"""
3D world rendering: textured floor, ceiling, and wall columns.

Consumes raycast output (from raycaster.cast_rays) and the textures dict
(from textures.generate_textures) and paints the perspective view.
"""

from __future__ import annotations

from typing import Any

import pygame
from shooter.constants import (
    WIDTH, HEIGHT, HALF_FOV, NUM_RAYS, TEX_SIZE,
    WHITE, CEIL, FLOOR,
)
from shooter.map import EXIT_TILE, BARRIER_TILE, DOOR_TILE, DOOR_POSITIONS, MAP_W, MAP_H
from shooter.types import DoorAnimMap, Textures, WallColumn
import numpy as np


# ---------------------------------------------------------------------------
# Floor / Ceiling
# ---------------------------------------------------------------------------
def draw_floor_ceiling(screen: pygame.Surface, px: float, py: float, pa: float,
                       textures: Textures | None,
                       horizon_offset: int = 0) -> None:
    """Render textured floor and ceiling."""
    horizon = HEIGHT // 2 + horizon_offset

    if textures is not None:
        _draw_fc_numpy(screen, px, py, pa, horizon, textures)
    else:
        pygame.draw.rect(screen, CEIL, (0, 0, WIDTH, horizon))
        pygame.draw.rect(screen, FLOOR, (0, horizon, WIDTH, HEIGHT - horizon))


def _draw_fc_numpy(screen: pygame.Surface, px: float, py: float, pa: float,
                   horizon: int, tex: Textures) -> None:
    """Fast floor/ceiling with numpy + surfarray (fully vectorized)."""
    pix = pygame.surfarray.pixels3d(screen)
    floor_t = tex['floor_np']
    ceil_t = tex['ceil_np']
    door_t = tex['door_np']
    half_h = HEIGHT * 0.5
    tm = TEX_SIZE

    angles = np.linspace(pa - HALF_FOV, pa + HALF_FOV, WIDTH, endpoint=False)
    cos_r = np.cos(angles)
    sin_r = np.sin(angles)

    # Ceiling mask: True for tiles that hold a door, so we can paint the
    # door-frame texture on the ceiling directly above each door.
    door_grid = np.zeros((MAP_W, MAP_H), dtype=bool)
    for c, r in DOOR_POSITIONS:
        door_grid[c, r] = True

    y_start = max(1, horizon + 1)
    if y_start < HEIGHT:
        rows = np.arange(y_start, HEIGHT)
        dists = half_h / (rows - horizon)
        shades = np.clip(1.0 - dists * 0.07, 0.12, 1.0)
        wx = (px + cos_r[:, np.newaxis] * dists[np.newaxis, :]) * tm
        wy = (py + sin_r[:, np.newaxis] * dists[np.newaxis, :]) * tm
        tx = wx.astype(np.int32) % tm
        ty = wy.astype(np.int32) % tm
        pix[:, y_start:HEIGHT, :] = (floor_t[tx, ty] * shades[np.newaxis, :, np.newaxis]).astype(np.uint8)

    y_end = min(HEIGHT, horizon)
    if y_end > 0:
        rows = np.arange(0, y_end)
        dists = half_h / (horizon - rows)
        shades = np.clip(1.0 - dists * 0.07, 0.12, 1.0)
        world_x = px + cos_r[:, np.newaxis] * dists[np.newaxis, :]
        world_y = py + sin_r[:, np.newaxis] * dists[np.newaxis, :]
        wx = world_x * tm
        wy = world_y * tm
        tx = wx.astype(np.int32) % tm
        ty = wy.astype(np.int32) % tm
        tile_c = np.clip(world_x.astype(np.int32), 0, MAP_W - 1)
        tile_r = np.clip(world_y.astype(np.int32), 0, MAP_H - 1)
        is_door_ceil = door_grid[tile_c, tile_r]
        # Darken the door texture a bit so it reads as an inset frame.
        ceil_samples = np.where(
            is_door_ceil[:, :, np.newaxis],
            door_t[tx, ty] * 0.65,
            ceil_t[tx, ty],
        )
        pix[:, :y_end, :] = (ceil_samples * shades[np.newaxis, :, np.newaxis]).astype(np.uint8)

    del pix


# ---------------------------------------------------------------------------
# 3D Walls
# ---------------------------------------------------------------------------
def draw_3d(screen: pygame.Surface, walls: list[WallColumn], z_buffer: list[float],
            font: Any, textures: Textures | None,
            horizon_offset: int = 0,
            door_anim: DoorAnimMap | None = None) -> None:
    """Render textured wall columns from raycast results.

    door_anim: optional dict of (col, row) -> {'progress': 0.0..1.0} for sliding doors.
    progress=0 is fully closed, progress=1 is fully open (door slid up into ceiling)."""
    horizon = HEIGHT // 2 + horizon_offset
    tex = textures

    col_w = max(WIDTH // NUM_RAYS, 1)
    sign_tiles: dict[int, list[int | None]] = {EXIT_TILE: [None, None]}
    for i, (depth, offset, side, hit_tile, bg_hit, tile_coords) in enumerate(walls):
        z_buffer[i] = depth
        x = i * col_w

        # Doors slide up on open/close — hide the top `progress` fraction of the column.
        door_progress = 0.0
        if hit_tile == DOOR_TILE and door_anim is not None:
            anim = door_anim.get(tile_coords)
            if anim is not None:
                door_progress = anim['progress']

        if hit_tile == EXIT_TILE:
            cols = tex['exit_cols'] if tex else None
        elif hit_tile == BARRIER_TILE:
            cols = tex['barrier_cols'] if tex else None
        elif hit_tile == DOOR_TILE:
            cols = tex['door_cols'] if tex else None
        else:
            cols = tex['wall_cols'] if tex else None

        # Draw the wall behind a barrier or an animating door so the top of the
        # doorway reveals the corridor beyond rather than the "infinite" ceiling.
        draw_bg = bg_hit is not None and (
            hit_tile == BARRIER_TILE or (hit_tile == DOOR_TILE and door_progress > 0)
        )
        if draw_bg and bg_hit is not None:
            bg_depth, bg_offset, bg_side, bg_tile = bg_hit
            bg_wall_h = min(int(HEIGHT / bg_depth), HEIGHT * 2)
            bg_shade = max(30, 255 - int(bg_depth * 18))
            if bg_side == 1:
                bg_shade = int(bg_shade * 0.7)
            bg_y = horizon - bg_wall_h // 2

            if tex:
                bg_cols = tex['exit_cols'] if bg_tile == EXIT_TILE else (tex['door_cols'] if bg_tile == DOOR_TILE else tex['wall_cols'])
                bg_tx = int(bg_offset * TEX_SIZE) % TEX_SIZE
                vis_top = max(0, bg_y)
                vis_bot = min(HEIGHT, bg_y + bg_wall_h)
                vis_h = vis_bot - vis_top
                if vis_h > 0:
                    scaled = pygame.transform.scale(bg_cols[bg_tx], (col_w + 1, bg_wall_h))
                    scaled.fill((bg_shade, bg_shade, bg_shade),
                                special_flags=pygame.BLEND_RGB_MULT)
                    screen.blit(scaled, (x, vis_top),
                                area=(0, vis_top - bg_y, col_w + 1, vis_h))
            else:
                bg_color = (bg_shade, bg_shade // 2 + 40, bg_shade // 3 + 20)
                pygame.draw.rect(screen, bg_color, (x, bg_y, col_w + 1, bg_wall_h))

        wall_h = min(int(HEIGHT / depth), HEIGHT * 2)
        shade = max(30, 255 - int(depth * 18))
        if side == 1:
            shade = int(shade * 0.7)

        if hit_tile == BARRIER_TILE:
            wall_h = wall_h // 3
            y = horizon + wall_h // 3
        else:
            y = horizon - wall_h // 2

        # Sliding door: the door retracts into the ceiling. Its top stays pinned
        # to the top of the doorway while the bottom rises, so the visible slice
        # is the TOP (1 - progress) of the doorway showing the BOTTOM of the door
        # texture. At progress=1 the door is entirely hidden in the ceiling.
        if door_progress > 0:
            visible_h = max(0, int((1.0 - door_progress) * wall_h))
            src_y_offset = wall_h - visible_h
        else:
            visible_h = wall_h
            src_y_offset = 0
        anim_bot = y + visible_h

        if cols and wall_h > 0:
            tx = int(offset * TEX_SIZE) % TEX_SIZE
            vis_top = max(0, y)
            vis_bot = min(HEIGHT, anim_bot)
            vis_h = vis_bot - vis_top
            if vis_h > 0:
                scaled = pygame.transform.scale(cols[tx], (col_w + 1, wall_h))
                scaled.fill((shade, shade, shade),
                            special_flags=pygame.BLEND_RGB_MULT)
                screen.blit(scaled, (x, vis_top),
                            area=(0, src_y_offset + (vis_top - y), col_w + 1, vis_h))
        else:
            color = (shade, shade // 2 + 40, shade // 3 + 20)
            pygame.draw.rect(screen, color, (x, y, col_w + 1, max(0, anim_bot - y)))

        if hit_tile in sign_tiles:
            if sign_tiles[hit_tile][0] is None:
                sign_tiles[hit_tile][0] = x
            sign_tiles[hit_tile][1] = x + col_w

    # draw signs on special tiles
    sign_config = {
        EXIT_TILE:  ("EXIT",  (20, 80, 20)),
    }
    for tile_type, (label, bg_color) in sign_config.items():
        left, right = sign_tiles[tile_type]
        if left is None or right is None:
            continue
        sign_w = right - left
        font_size = max(8, min(int(sign_w * 0.5), 60))
        text_surf, text_rect = font.render(label, WHITE, size=font_size)
        tx = left + sign_w // 2 - text_rect.width // 2
        ty = horizon - text_rect.height // 2
        bg_rect = pygame.Rect(tx - 4, ty - 2, text_rect.width + 8, text_rect.height + 4)
        pygame.draw.rect(screen, bg_color, bg_rect)
        pygame.draw.rect(screen, WHITE, bg_rect, 1)
        screen.blit(text_surf, (tx, ty))
