from typing import Dict
from string import ascii_uppercase
from battleship.ai.base_ai import BattleshipAI
from battleship.robot.ot2_utils import OT2Manager
from battleship.plate_state_processor import DualPlateStateProcessor # A new processor for two plates
from typing import Any, List

class BattleshipGame:
    """Manages a competitive game of Battleship between two AI players."""

    def __init__(self,
                 player_1_ai: BattleshipAI,
                 player_2_ai: BattleshipAI,
                 plate_processor: DualPlateStateProcessor,
                 robot: OT2Manager):
        self.players = {'player_1': player_1_ai, 'player_2': player_2_ai}
        self.plate_processor = plate_processor
        self.robot = robot
        self.history: List[Dict[str, Any]] = []

    def run_game_live(self):
        """
        Main game loop that yields the game state after each individual move.
        This is designed for use with live front-end updates.
        """
        print("--- BATTLESHIP COMPETITION START (LIVE) ---")
        print(f"Player 1: {self.players['player_1'].__class__.__name__}")
        print(f"Player 2: {self.players['player_2'].__class__.__name__}")
        print("------------------------------------")

        turn = 0
        while True:
            turn += 1
            
            for player_id in ['player_1', 'player_2']:
                ai = self.players[player_id]
                
                # 1. Get move from the current player's AI
                move = ai.select_next_move()
                
                # 2. Fire the missile on the physical plate
                well_name = f"{ascii_uppercase[move[0]]}{move[1] + 1}"
                print(f"Turn {turn}, {player_id}: Firing at {well_name}...")
                self.robot.add_fire_missile_action(well_name, plate_slot=1 if player_id == 'player_1' else 2)
                self.robot.execute_actions_on_remote()

                # 3. Determine the result from the camera
                result = self.plate_processor.determine_well_state(plate_id=player_id, well=move)
                print(f"Result: {result.name}!")
                
                # 4. Update the AI with the result and log history
                ai.record_shot_result(move, result)
                self.history.append({
                    'turn': turn,
                    'player': player_id,
                    'move': well_name,
                    'result': result.name
                })

                # 5. Yield the complete current state for the UI
                current_state = {
                    'turn': turn,
                    'active_player': player_id,
                    'move': well_name,
                    'result': result.name,
                    'board_p1': self.players['player_1'].board_state,
                    'board_p2': self.players['player_2'].board_state,
                    'history': self.history,
                    'winner': None
                }
                yield current_state

                # 6. Check for a winner
                if ai.has_won():
                    print(f"\n--- GAME OVER ---")
                    print(f"🎉 {player_id} ({ai.__class__.__name__}) has sunk all ships and wins in {turn} turns! 🎉")
                    current_state['winner'] = player_id
                    yield current_state
                    return # End the generator

    def run_game(self):
        """Main game loop that alternates turns until a winner is found."""
        print("--- BATTLESHIP COMPETITION START ---")
        print(f"Player 1: {self.players['player_1'].__class__.__name__}")
        print(f"Player 2: {self.players['player_2'].__class__.__name__}")
        print("------------------------------------")

        turn = 0
        while True:
            turn += 1
            print(f"\n--- Turn {turn} ---")
            
            for player_id, opponent_id in [('player_1', 'player_2'), ('player_2', 'player_1')]:
                # 1. Get move from the current player's AI
                ai = self.players[player_id]
                move = ai.select_next_move()
                
                # 2. Fire the missile on the physical plate
                well_name = f"{ascii_uppercase[move[0]]}{move[1] + 1}"
                print(f"{ai.player_id} ({ai.__class__.__name__}) fires at {well_name}...")
                self.robot.add_fire_missile_action(well_name, plate_slot=1 if player_id == 'player_1' else 2)
                self.robot.execute_actions_on_remote()

                # 3. Determine the result from the camera
                result = self.plate_processor.determine_well_state(plate_id=player_id, well=move)
                print(f"Result: {result.name}!")
                self.history.append({
                    'turn': turn,
                    'player_id': player_id,
                    'move': move,
                    'result': result
                })
                # 4. Update the AI with the result
                ai.record_shot_result(move, result)

                # 5. Check for a winner
                if ai.has_won():
                    print(f"\n--- GAME OVER ---")
                    print(f"🎉 Player {player_id} ({ai.__class__.__name__}) has sunk all ships and wins in {turn} turns! 🎉")
                    return player_id