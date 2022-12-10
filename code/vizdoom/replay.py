#!/usr/bin/env python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#####################################################################

import os
from random import choice
import vizdoom as vzd
import cv2

game = vzd.DoomGame()
game.set_screen_format(vzd.ScreenFormat.BGR24)
# Use other config file if you wish.
game.load_config("maps/D3_battle.cfg")
game.set_episode_timeout(100)

# Record episodes while playing in 320x240 resolution without HUD
#game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
game.set_render_hud(True)

# Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(vzd.Mode.PLAYER)


# OpenCV uses a BGR colorspace by default.
game.set_screen_format(vzd.ScreenFormat.BGR24)

# game.set_screen_format(ScreenFormat.RGB24)
# game.set_screen_format(ScreenFormat.RGBA32)
# game.set_screen_format(ScreenFormat.ARGB32)
# game.set_screen_format(ScreenFormat.BGRA32)
# game.set_screen_format(ScreenFormat.ABGR32)
# game.set_screen_format(ScreenFormat.GRAY8)

# Raw Doom buffer with palette's values. This one makes no sense in particular
# game.set_screen_format(ScreenFormat.DOOM_256_COLORS)

# Sets resolution for all buffers.
game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

game.set_render_hud(True)
game.set_render_minimal_hud(False)

game.set_mode(vzd.Mode.SPECTATOR)
game.init()

writer = cv2.VideoWriter('D3_battle.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (800, 600))
print("\nREPLAY OF EPISODE")
print("************************\n")


for i in range(0,2):
    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode("Lmps/episode_{}.lmp".format(i))

    while not game.is_episode_finished():
        s = game.get_state()

        # Use advance_action instead of make_action.
        game.advance_action()
        screen = s.screen_buffer
        print(screen.shape)
        writer.write(screen)

        r = game.get_last_reward()
        # game.get_last_action is not supported and don't work for replay at the moment.

        print("State #" + str(s.number))
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")
writer.release()
print("Episode", "finished.")
print("total reward:", game.get_total_reward())
print("************************")

game.close()
