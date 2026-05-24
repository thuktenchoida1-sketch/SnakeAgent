import pygame
import random
from enum import Enum
from collections import namedtuple
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from model import Linear_QNet

pygame.init()
FONT_FILE = Path(__file__).resolve().parent / 'arial.ttf'
font = pygame.font.Font(str(FONT_FILE), 25)
title_font = pygame.font.Font(str(FONT_FILE), 34)
small_font = pygame.font.Font(str(FONT_FILE), 18)
game_over_font = pygame.font.Font(str(FONT_FILE), 58)
button_font = pygame.font.Font(str(FONT_FILE), 23)
menu_title_font = pygame.font.Font(str(FONT_FILE), 54)
#font = pygame.font.SysFont('arial', 25)


#reset
#rewar
#play(action - direction)
#game_iteration
# is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (244, 248, 255)
RED = (239, 83, 80)
BLUE1 = (56, 150, 230)
BLUE2 = (134, 217, 255)
BLACK = (7, 10, 16)
DARK = (10, 14, 21)
PANEL = (18, 24, 34)
PANEL_ALT = (23, 31, 43)
PANEL_EDGE = (61, 75, 99)
PANEL_EDGE_SOFT = (36, 47, 64)
BOARD_BG = (6, 10, 16)
GRID = (27, 36, 50)
GRID_SOFT = (17, 24, 34)
GREEN1 = (0, 188, 125)
GREEN2 = (112, 238, 183)
MUTED = (160, 174, 196)
CYAN = (116, 232, 255)
BUTTON_TOP = (27, 43, 60)
BUTTON_BOTTOM = (18, 28, 41)
BUTTON_EDGE = (77, 196, 230)
BUTTON_HOVER = (35, 59, 78)
OVERLAY_RED = (156, 45, 56)
SHADOW = (0, 0, 0, 90)

BLOCK_SIZE = 20
SPEED = 40
HUMAN_SPEED = 14
TOP_BAR_HEIGHT = 76
BOTTOM_MARGIN = 22
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pth"

class SnakeGameAI:
    
    def __init__(self, w=640, h=480, display=None):
        self.w = w
        self.h = h
        self.display = display
        if self.display is None:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
        
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point((self.w // (2 * BLOCK_SIZE)) * BLOCK_SIZE,
                          (self.h // (2 * BLOCK_SIZE)) * BLOCK_SIZE)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action, update_ui=True, handle_events=True):
        self.frame_iteration += 1
        # 1. collect user input
        if handle_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        if update_ui:
            self._update_ui()
            self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(DARK)
        board = pygame.Rect(0, 0, self.w, self.h)
        _draw_board_background(self.display, board)
        
        for pt in self.snake:
            cell = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE).inflate(-3, -3)
            pygame.draw.rect(self.display, BLUE1, cell, border_radius=5)
            pygame.draw.rect(self.display, BLUE2, cell.inflate(-8, -8), border_radius=3)
            
        food = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE).inflate(-3, -3)
        pygame.draw.rect(self.display, RED, food, border_radius=6)
        pygame.draw.rect(self.display, (255, 151, 116), food.inflate(-8, -8), border_radius=4)
        
        score_text = small_font.render("Score " + str(self.score), True, WHITE)
        score_rect = score_text.get_rect(topleft=(12, 10)).inflate(22, 12)
        pygame.draw.rect(self.display, PANEL, score_rect, border_radius=8)
        pygame.draw.rect(self.display, PANEL_EDGE_SOFT, score_rect, 1, border_radius=8)
        self.display.blit(score_text, score_text.get_rect(center=score_rect.center))
        pygame.display.flip()

    def draw(self, surface, rect, title, subtitle, primary, secondary, high_score=0):
        _draw_panel(surface, rect)

        header_h = 58
        board = pygame.Rect(rect.x + 20, rect.y + header_h, self.w, self.h)
        header = pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, header_h - 2)
        pygame.draw.rect(surface, PANEL_ALT, header, border_radius=10)
        pygame.draw.line(surface, PANEL_EDGE_SOFT, (rect.x + 16, rect.y + header_h - 1), (rect.right - 16, rect.y + header_h - 1))
        _draw_board_background(surface, board)

        for pt in self.snake:
            cell = pygame.Rect(board.x + pt.x, board.y + pt.y, BLOCK_SIZE, BLOCK_SIZE).inflate(-3, -3)
            pygame.draw.rect(surface, primary, cell, border_radius=5)
            pygame.draw.rect(surface, secondary, cell.inflate(-8, -8), border_radius=3)

        food = pygame.Rect(board.x + self.food.x, board.y + self.food.y, BLOCK_SIZE, BLOCK_SIZE).inflate(-3, -3)
        pygame.draw.rect(surface, RED, food, border_radius=7)
        pygame.draw.rect(surface, (255, 151, 116), food.inflate(-8, -8), border_radius=4)

        score_text = font.render("Score " + str(self.score), True, WHITE)
        score_pill = score_text.get_rect().inflate(28, 14)
        score_pill.topright = (rect.right - 18, rect.y + 13)
        pygame.draw.rect(surface, (14, 20, 29), score_pill, border_radius=8)
        pygame.draw.rect(surface, PANEL_EDGE_SOFT, score_pill, 1, border_radius=8)
        surface.blit(score_text, score_text.get_rect(center=score_pill.center))

        best_score = max(high_score, self.score)
        best_text = small_font.render("Best " + str(best_score), True, CYAN)
        best_pill = best_text.get_rect().inflate(22, 12)
        best_pill.centery = score_pill.centery
        best_pill.right = score_pill.left - 8
        pygame.draw.rect(surface, (14, 20, 29), best_pill, border_radius=8)
        pygame.draw.rect(surface, PANEL_EDGE_SOFT, best_pill, 1, border_radius=8)
        surface.blit(best_text, best_text.get_rect(center=best_pill.center))

        title_max_w = max(80, best_pill.left - rect.x - 42)
        title_text = _render_fit(title, title_font, WHITE, title_max_w)
        subtitle_text = _render_fit(subtitle, small_font, MUTED, title_max_w)
        surface.blit(title_text, (rect.x + 22, rect.y + 8))
        surface.blit(subtitle_text, (rect.x + 24, rect.y + 39))

    def _move(self, action):
        # straigt right,left 

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir



        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)


class HumanSnakeGame(SnakeGameAI):
    def move_absolute(self, direction):
        opposites = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
        }
        if direction != opposites[self.direction]:
            self.direction = direction

    def play_human_step(self):
        self.frame_iteration += 1
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        self.snake.insert(0, self.head)

        game_over = self.is_collision()
        if game_over:
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return game_over, self.score


class DemoAgent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)
        self.loaded_model = False
        self.reload_model()
        self.model.eval()

    def reload_model(self):
        self.loaded_model = False
        if not MODEL_PATH.exists():
            return

        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
        self.model.load_state_dict(state_dict)
        self.loaded_model = True
        self.model.eval()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        danger = state[:3].astype(bool)
        if not self.loaded_model:
            return self._safe_food_action(state, danger)

        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state0)

        safe_scores = prediction.clone()
        for idx, is_dangerous in enumerate(danger):
            if is_dangerous:
                safe_scores[idx] = float('-inf')

        move = torch.argmax(safe_scores if torch.isfinite(safe_scores).any() else prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def _safe_food_action(self, state, danger):
        directions = {
            Direction.LEFT: np.array([-1, 0]),
            Direction.RIGHT: np.array([1, 0]),
            Direction.UP: np.array([0, -1]),
            Direction.DOWN: np.array([0, 1]),
        }
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current = Direction.LEFT if state[3] else Direction.RIGHT if state[4] else Direction.UP if state[5] else Direction.DOWN
        idx = clock_wise.index(current)
        candidates = [
            (0, clock_wise[idx]),
            (1, clock_wise[(idx + 1) % 4]),
            (2, clock_wise[(idx - 1) % 4]),
        ]
        food_vector = np.array([
            -1 if state[7] else 1 if state[8] else 0,
            -1 if state[9] else 1 if state[10] else 0,
        ])

        best_move = 0
        best_score = -999
        for move, direction in candidates:
            if danger[move]:
                continue
            score = int(np.dot(directions[direction], food_vector))
            if score > best_score:
                best_score = score
                best_move = move

        final_move = [0, 0, 0]
        final_move[best_move] = 1
        return final_move


def _fit_board_size(screen_w, screen_h):
    available_w = (screen_w - 148) // 2
    available_h = screen_h - TOP_BAR_HEIGHT - BOTTOM_MARGIN - 78
    board_w = max(BLOCK_SIZE * 12, (available_w // BLOCK_SIZE) * BLOCK_SIZE)
    board_h = max(BLOCK_SIZE * 12, (available_h // BLOCK_SIZE) * BLOCK_SIZE)
    return board_w, board_h


def _fit_single_board_size(screen_w, screen_h):
    available_w = screen_w - 96
    available_h = screen_h - TOP_BAR_HEIGHT - BOTTOM_MARGIN - 78
    board_w = max(BLOCK_SIZE * 16, (available_w // BLOCK_SIZE) * BLOCK_SIZE)
    board_h = max(BLOCK_SIZE * 16, (available_h // BLOCK_SIZE) * BLOCK_SIZE)
    return board_w, board_h


def _panel_top(screen_h, panel_h):
    available_h = screen_h - TOP_BAR_HEIGHT - BOTTOM_MARGIN
    return TOP_BAR_HEIGHT + max(0, (available_h - panel_h) // 2)


def _open_fullscreen(caption):
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption(caption)
    return screen, screen.get_width(), screen.get_height()


def _render_fit(text, font_obj, color, max_width):
    rendered = font_obj.render(text, True, color)
    if rendered.get_width() <= max_width:
        return rendered
    scale = max(0.55, max_width / max(1, rendered.get_width()))
    width = max(1, int(rendered.get_width() * scale))
    height = max(1, int(rendered.get_height() * scale))
    return pygame.transform.smoothscale(rendered, (width, height))


def _draw_app_background(surface):
    surface.fill(DARK)
    width, height = surface.get_size()
    for x in range(0, width, 40):
        pygame.draw.line(surface, GRID_SOFT, (x, 0), (x, height))
    for y in range(0, height, 40):
        pygame.draw.line(surface, GRID_SOFT, (0, y), (width, y))

    top_band = pygame.Surface((width, min(96, height)), pygame.SRCALPHA)
    top_band.fill((255, 255, 255, 8))
    surface.blit(top_band, (0, 0))


def _draw_panel(surface, rect):
    shadow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow, SHADOW, shadow.get_rect(), border_radius=10)
    surface.blit(shadow, (rect.x, rect.y + 3))
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_EDGE, rect, 1, border_radius=10)


def _draw_board_background(surface, board):
    pygame.draw.rect(surface, BOARD_BG, board, border_radius=6)
    for x in range(board.left, board.right + 1, BLOCK_SIZE):
        pygame.draw.line(surface, GRID, (x, board.top), (x, board.bottom))
    for y in range(board.top, board.bottom + 1, BLOCK_SIZE):
        pygame.draw.line(surface, GRID, (board.left, y), (board.right, y))
    pygame.draw.rect(surface, PANEL_EDGE_SOFT, board, 1, border_radius=6)


def _game_over_button_rects(panel, allow_restart=True):
    gap = max(10, min(18, panel.w // 32))
    button_count = 3 if allow_restart else 2
    button_w = max(82, min(140, (panel.w - 64 - gap * (button_count - 1)) // button_count))
    button_h = 48
    total_w = button_w * button_count + gap * (button_count - 1)
    y = panel.bottom - button_h - 28
    left = panel.centerx - total_w // 2
    restart = pygame.Rect(left, y, button_w, button_h) if allow_restart else None
    menu_x = restart.right + gap if restart else left
    menu_btn = pygame.Rect(menu_x, y, button_w, button_h)
    quit_btn = pygame.Rect(menu_btn.right + gap, y, button_w, button_h)
    return restart, menu_btn, quit_btn


def _draw_button(surface, rect, label, hover=False):
    rect = rect.copy()
    if hover:
        rect.inflate_ip(4, 4)

    shadow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow, SHADOW, shadow.get_rect(), border_radius=8)
    surface.blit(shadow, (rect.x, rect.y + 2))

    fill = BUTTON_HOVER if hover else BUTTON_TOP
    pygame.draw.rect(surface, fill, rect, border_radius=8)
    pygame.draw.rect(surface, BUTTON_EDGE if hover else PANEL_EDGE, rect, 1, border_radius=8)
    accent = pygame.Rect(rect.x + 10, rect.y + 7, max(0, rect.w - 20), 2)
    pygame.draw.rect(surface, BUTTON_EDGE, accent, border_radius=1)
    label_text = _render_fit(label, button_font, CYAN, rect.w - 22)
    surface.blit(label_text, label_text.get_rect(center=rect.center))


def _save_screenshot(surface):
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    filename = SCREENSHOT_DIR / f"snake_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    pygame.image.save(surface, filename)
    return filename


def _draw_top_bar(surface, notice=None):
    menu_rect = pygame.Rect(22, 18, 118, 42)
    _draw_button(surface, menu_rect, "Menu", menu_rect.collidepoint(pygame.mouse.get_pos()))

    if notice:
        text = _render_fit(notice, small_font, MUTED, max(60, surface.get_width() - menu_rect.right - 44))
        surface.blit(text, (menu_rect.right + 18, menu_rect.centery - text.get_height() // 2))

    return menu_rect


def _draw_menu_background(surface, stars):
    _draw_app_background(surface)
    width, height = surface.get_size()
    band = pygame.Surface((width, max(120, height // 5)), pygame.SRCALPHA)
    band.fill((255, 255, 255, 7))
    surface.blit(band, (0, height // 2 - band.get_height() // 2))


def show_mode_menu():
    screen, screen_w, screen_h = _open_fullscreen('Snake: Mode Selection')
    clock = pygame.time.Clock()
    screenshot_notice_until = 0
    stars = []

    margin = max(24, min(screen_w, screen_h) // 24)
    panel_w = min(560, screen_w - margin * 2)
    button_w = min(380, panel_w - 64)
    button_h = max(46, min(56, screen_h // 18))
    gap = max(12, min(18, screen_h // 70))
    button_stack_h = button_h * 4 + gap * 3
    panel_h = min(screen_h - margin * 2, max(360, button_stack_h + 190))
    panel = pygame.Rect(0, 0, panel_w, panel_h)
    panel.center = (screen_w // 2, screen_h // 2)
    start_y = panel.bottom - button_stack_h - 34
    buttons = [
        ("human", "Human Version", pygame.Rect(panel.centerx - button_w // 2, start_y, button_w, button_h)),
        ("agent", "Agentic AI", pygame.Rect(panel.centerx - button_w // 2, start_y + button_h + gap, button_w, button_h)),
        ("both", "AI vs Human", pygame.Rect(panel.centerx - button_w // 2, start_y + (button_h + gap) * 2, button_w, button_h)),
    ]
    exit_rect = pygame.Rect(panel.centerx - button_w // 2, start_y + (button_h + gap) * 3, button_w, button_h)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return None
                if event.key == pygame.K_p:
                    _save_screenshot(screen)
                    screenshot_notice_until = pygame.time.get_ticks() + 1600
                if event.key == pygame.K_1:
                    return "human"
                if event.key == pygame.K_2:
                    return "agent"
                if event.key == pygame.K_3:
                    return "both"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for mode, _, rect in buttons:
                    if rect.collidepoint(event.pos):
                        return mode
                if exit_rect.collidepoint(event.pos):
                    pygame.quit()
                    return None

        _draw_menu_background(screen, stars)
        _draw_panel(screen, panel)
        title_1 = _render_fit("SNAKE AI", menu_title_font, WHITE, panel.w - 72)
        title_2 = _render_fit("Mode Menu", title_font, CYAN, panel.w - 72)
        screen.blit(title_1, title_1.get_rect(center=(panel.centerx, panel.y + 64)))
        screen.blit(title_2, title_2.get_rect(center=(panel.centerx, panel.y + 118)))

        hint = _render_fit("Choose a mode to start", small_font, MUTED, panel.w - 72)
        screen.blit(hint, hint.get_rect(center=(panel.centerx, panel.y + 154)))
        pygame.draw.line(screen, PANEL_EDGE_SOFT, (panel.x + 32, start_y - 22), (panel.right - 32, start_y - 22))

        mouse_pos = pygame.mouse.get_pos()
        for _, label, rect in buttons:
            _draw_button(screen, rect, label, rect.collidepoint(mouse_pos))
        _draw_button(screen, exit_rect, "Quit", exit_rect.collidepoint(mouse_pos))
        if pygame.time.get_ticks() < screenshot_notice_until:
            notice = small_font.render("Screenshot saved", True, MUTED)
            screen.blit(notice, notice.get_rect(center=(panel.centerx, min(screen_h - 24, exit_rect.bottom + 30))))

        pygame.display.flip()
        clock.tick(60)


def draw_game_over_overlay(surface, panel, score, label, allow_restart=True):
    board = pygame.Rect(panel.x + 20, panel.y + 58, panel.w - 40, panel.h - 78)
    dim = pygame.Surface((board.w, board.h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 150))
    surface.blit(dim, board)

    modal_w = min(520, max(240, board.w - 48))
    modal_h = min(260, max(210, board.h - 48))
    modal = pygame.Rect(0, 0, modal_w, modal_h)
    modal.center = board.center
    _draw_panel(surface, modal)

    accent = pygame.Rect(modal.x + 22, modal.y + 20, modal.w - 44, 3)
    pygame.draw.rect(surface, OVERLAY_RED, accent, border_radius=2)

    over_text = _render_fit("Game Over", game_over_font, (255, 142, 142), modal.w - 48)
    surface.blit(over_text, over_text.get_rect(center=(modal.centerx, modal.y + 78)))

    score_text = _render_fit(f"{label} Score: {score}", font, WHITE, modal.w - 48)
    surface.blit(score_text, score_text.get_rect(center=(modal.centerx, modal.y + 128)))

    restart, menu_btn, quit_btn = _game_over_button_rects(modal, allow_restart)
    mouse_pos = pygame.mouse.get_pos()
    if restart:
        _draw_button(surface, restart, "Restart", restart.collidepoint(mouse_pos))
    _draw_button(surface, menu_btn, "Menu", menu_btn.collidepoint(mouse_pos))
    _draw_button(surface, quit_btn, "Quit", quit_btn.collidepoint(mouse_pos))
    return restart, menu_btn, quit_btn


def run_single_screen(mode):
    caption = 'Snake: Human' if mode == "human" else 'Snake: Agentic AI'
    screen, screen_w, screen_h = _open_fullscreen(caption)
    clock = pygame.time.Clock()

    board_w, board_h = _fit_single_board_size(screen_w, screen_h)
    panel_w = board_w + 40
    panel_h = board_h + 78
    panel = pygame.Rect((screen_w - panel_w) // 2, _panel_top(screen_h, panel_h), panel_w, panel_h)

    if mode == "human":
        game = HumanSnakeGame(board_w, board_h, display=screen)
        agent = None
        title = "Your Snake"
        subtitle = "Arrow keys or WASD"
        primary = GREEN1
        secondary = GREEN2
        score_label = "Your"
        tick_rate = HUMAN_SPEED
    else:
        game = SnakeGameAI(board_w, board_h, display=screen)
        agent = DemoAgent()
        title = "Agent Snake"
        subtitle = "Autopilot model"
        primary = BLUE1
        secondary = BLUE2
        score_label = "Agent"
        tick_rate = SPEED

    done = False
    high_score = 0
    step_tick = 0
    overlay_buttons = None
    menu_button = None
    screenshot_notice_until = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return None
                if event.key == pygame.K_p:
                    _save_screenshot(screen)
                    screenshot_notice_until = pygame.time.get_ticks() + 1600
                if mode == "human" and not done:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        game.move_absolute(Direction.UP)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        game.move_absolute(Direction.DOWN)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        game.move_absolute(Direction.LEFT)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        game.move_absolute(Direction.RIGHT)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if menu_button and menu_button.collidepoint(event.pos):
                    return "menu"
                if done and overlay_buttons:
                    restart, menu_btn, quit_btn = overlay_buttons
                    if restart and restart.collidepoint(event.pos):
                        if agent:
                            agent.reload_model()
                        high_score = max(high_score, game.score)
                        game.reset()
                        done = False
                        step_tick = 0
                    elif menu_btn.collidepoint(event.pos):
                        return "menu"
                    elif quit_btn.collidepoint(event.pos):
                        pygame.quit()
                        return None

        step_tick += 1
        if not done and step_tick >= max(1, SPEED // tick_rate):
            if mode == "human":
                done, _ = game.play_human_step()
                if done:
                    high_score = max(high_score, game.score)
            else:
                state = agent.get_state(game)
                action = agent.get_action(state)
                _, done, _ = game.play_step(action, update_ui=False, handle_events=False)
                if done:
                    high_score = max(high_score, game.score)
                    game.reset()
                    done = False
            step_tick = 0

        _draw_app_background(screen)
        game.draw(screen, panel, title, subtitle, primary, secondary, high_score)
        # Only show the game-over overlay for human mode; hide for agent mode
        overlay_buttons = draw_game_over_overlay(screen, panel, game.score, score_label, allow_restart=(mode == "human")) if done and mode == "human" else None
        notice = "Screenshot saved" if pygame.time.get_ticks() < screenshot_notice_until else None
        menu_button = _draw_top_bar(screen, notice)
        pygame.display.flip()
        clock.tick(SPEED)


def run_split_screen():
    screen, screen_w, screen_h = _open_fullscreen('Snake: AI vs Human')
    clock = pygame.time.Clock()

    board_w, board_h = _fit_board_size(screen_w, screen_h)
    panel_w = board_w + 40
    panel_h = board_h + 78
    top = _panel_top(screen_h, panel_h)
    left_panel = pygame.Rect(20, top, panel_w, panel_h)
    right_panel = pygame.Rect(screen_w - panel_w - 20, top, panel_w, panel_h)
    divider_x = screen_w // 2

    ai_game = SnakeGameAI(board_w, board_h, display=screen)
    human_game = HumanSnakeGame(board_w, board_h, display=screen)
    agent = DemoAgent()
    human_done = False
    ai_done = False
    ai_high_score = 0
    human_high_score = 0
    human_tick = 0
    ai_buttons = None
    human_buttons = None
    menu_button = None
    screenshot_notice_until = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return None
                if event.key == pygame.K_p:
                    _save_screenshot(screen)
                    screenshot_notice_until = pygame.time.get_ticks() + 1600
                if event.key == pygame.K_r and human_done:
                    human_game.reset()
                    human_done = False
                if event.key in (pygame.K_UP, pygame.K_w):
                    human_game.move_absolute(Direction.UP)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    human_game.move_absolute(Direction.DOWN)
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    human_game.move_absolute(Direction.LEFT)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    human_game.move_absolute(Direction.RIGHT)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                if menu_button and menu_button.collidepoint(mouse_pos):
                    return "menu"
                if ai_done and ai_buttons:
                    restart, menu_btn, quit_btn = ai_buttons
                    if restart and restart.collidepoint(mouse_pos):
                        agent.reload_model()
                        ai_high_score = max(ai_high_score, ai_game.score)
                        ai_game.reset()
                        ai_done = False
                    elif menu_btn.collidepoint(mouse_pos):
                        return "menu"
                    elif quit_btn.collidepoint(mouse_pos):
                        pygame.quit()
                        return None
                if human_done and human_buttons:
                    restart, menu_btn, quit_btn = human_buttons
                    if restart.collidepoint(mouse_pos):
                        human_high_score = max(human_high_score, human_game.score)
                        human_game.reset()
                        human_done = False
                    elif menu_btn.collidepoint(mouse_pos):
                        return "menu"
                    elif quit_btn.collidepoint(mouse_pos):
                        pygame.quit()
                        return None

        if not ai_done:
            ai_state = agent.get_state(ai_game)
            ai_action = agent.get_action(ai_state)
            _, ai_done, _ = ai_game.play_step(ai_action, update_ui=False, handle_events=False)
            if ai_done:
                ai_high_score = max(ai_high_score, ai_game.score)
                ai_game.reset()
                ai_done = False

        human_tick += 1
        if not human_done and human_tick >= max(1, SPEED // HUMAN_SPEED):
            human_done, _ = human_game.play_human_step()
            if human_done:
                human_high_score = max(human_high_score, human_game.score)
            human_tick = 0

        _draw_app_background(screen)
        pygame.draw.line(screen, PANEL_EDGE_SOFT, (divider_x, TOP_BAR_HEIGHT), (divider_x, screen_h - BOTTOM_MARGIN), 1)
        ai_game.draw(screen, left_panel, "Agent Snake", "Autopilot model", BLUE1, BLUE2, ai_high_score)
        human_game.draw(screen, right_panel, "Your Snake", "Arrow keys or WASD", GREEN1, GREEN2, human_high_score)
        # Do not draw a game-over overlay for the AI/agent panel.
        ai_buttons = None
        human_buttons = draw_game_over_overlay(screen, right_panel, human_game.score, "Your", allow_restart=True) if human_done else None
        notice = "Screenshot saved" if pygame.time.get_ticks() < screenshot_notice_until else None
        menu_button = _draw_top_bar(screen, notice)
        pygame.display.flip()
        clock.tick(SPEED)


def run_from_menu():
    while True:
        mode = show_mode_menu()
        if mode == "human":
            result = run_single_screen("human")
        elif mode == "agent":
            result = run_single_screen("agent")
        elif mode == "both":
            result = run_split_screen()
        else:
            return

        if result != "menu":
            return


if __name__ == '__main__':
    run_from_menu()
