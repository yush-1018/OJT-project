#!/usr/bin/env python3
"""
Lightweight Probability-Guided Number Guessing Simulator
- Multiple strategies (binary, probability, hybrid, random)
- Play a single game (manual or auto)
- Auto-play multiple games and see stats
"""

import random
import time
import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any


# ============================================================================
# CONFIGURATION
# ============================================================================

class StrategyType(Enum):
    BINARY = "binary"
    PROBABILITY = "probability"
    HYBRID = "hybrid"
    RANDOM = "random"


@dataclass
class Configuration:
    min_range: int = 1
    max_range: int = 100
    max_guesses: int = 10
    strategy: str = "hybrid"

    def validate(self) -> List[str]:
        errors = []
        if self.min_range < 1:
            errors.append("min_range must be >= 1")
        if self.max_range <= self.min_range:
            errors.append("max_range must be > min_range")
        if self.max_range > 1_000_000:
            errors.append("max_range must be <= 1,000,000")
        if self.max_guesses < 1:
            errors.append("max_guesses must be >= 1")
        if self.max_guesses > 1000:
            errors.append("max_guesses must be <= 1000")
        if self.strategy not in [s.value for s in StrategyType]:
            errors.append(f"strategy must be one of {[s.value for s in StrategyType]}")
        return errors


# ============================================================================
# STRATEGIES
# ============================================================================

class GuessingStrategy:
    def __init__(self, min_val: int, max_val: int):
        self.min_val = min_val
        self.max_val = max_val
        self.history: List[Tuple[int, str]] = []

    def make_guess(self) -> int:
        raise NotImplementedError

    def update(self, guess: int, result: str):
        self.history.append((guess, result))


class BinarySearchStrategy(GuessingStrategy):
    def __init__(self, min_val: int, max_val: int):
        super().__init__(min_val, max_val)
        self.low = min_val
        self.high = max_val

    def make_guess(self) -> int:
        return (self.low + self.high) // 2

    def update(self, guess: int, result: str):
        super().update(guess, result)
        if result == "higher":
            self.low = guess + 1
        elif result == "lower":
            self.high = guess - 1


class ProbabilityGuidedStrategy(GuessingStrategy):
    def __init__(self, min_val: int, max_val: int):
        super().__init__(min_val, max_val)
        self.probabilities = [1.0] * (max_val - min_val + 1)

    def make_guess(self) -> int:
        total = sum(self.probabilities)
        if total <= 0:
            return random.randint(self.min_val, self.max_val)

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(self.probabilities):
            cumulative += p / total
            if r <= cumulative:
                return self.min_val + i
        return self.max_val

    def update(self, guess: int, result: str):
        super().update(guess, result)
        idx = guess - self.min_val
        if not (0 <= idx < len(self.probabilities)):
            return
        if result == "higher":
            for i in range(idx + 1):
                self.probabilities[i] = 0.0
        elif result == "lower":
            for i in range(idx, len(self.probabilities)):
                self.probabilities[i] = 0.0


class HybridStrategy(GuessingStrategy):
    def __init__(self, min_val: int, max_val: int):
        super().__init__(min_val, max_val)
        self.binary = BinarySearchStrategy(min_val, max_val)
        self.prob = ProbabilityGuidedStrategy(min_val, max_val)
        self.use_prob_chance = 0.3

    def make_guess(self) -> int:
        if random.random() < self.use_prob_chance:
            return self.prob.make_guess()
        return self.binary.make_guess()

    def update(self, guess: int, result: str):
        super().update(guess, result)
        self.binary.update(guess, result)
        self.prob.update(guess, result)


class RandomStrategy(GuessingStrategy):
    def __init__(self, min_val: int, max_val: int):
        super().__init__(min_val, max_val)
        self.remaining = set(range(min_val, max_val + 1))

    def make_guess(self) -> int:
        if not self.remaining:
            return random.randint(self.min_val, self.max_val)
        return random.choice(list(self.remaining))

    def update(self, guess: int, result: str):
        super().update(guess, result)
        self.remaining.discard(guess)
        if result == "higher":
            self.remaining = {x for x in self.remaining if x > guess}
        elif result == "lower":
            self.remaining = {x for x in self.remaining if x < guess}


# ============================================================================
# GAME ENGINE
# ============================================================================

class GameEngine:
    def __init__(self, config: Configuration):
        self.config = config
        self.target: Optional[int] = None
        self.strategy: Optional[GuessingStrategy] = None
        self.guess_count: int = 0

    def _create_strategy(self) -> GuessingStrategy:
        s = self.config.strategy
        if s == StrategyType.BINARY.value:
            return BinarySearchStrategy(self.config.min_range, self.config.max_range)
        if s == StrategyType.PROBABILITY.value:
            return ProbabilityGuidedStrategy(self.config.min_range, self.config.max_range)
        if s == StrategyType.HYBRID.value:
            return HybridStrategy(self.config.min_range, self.config.max_range)
        if s == StrategyType.RANDOM.value:
            return RandomStrategy(self.config.min_range, self.config.max_range)
        raise ValueError(f"Unknown strategy: {s}")

    def start_game(self):
        self.target = random.randint(self.config.min_range, self.config.max_range)
        self.strategy = self._create_strategy()
        self.guess_count = 0

    def make_guess(self, guess: Optional[int] = None) -> Dict[str, Any]:
        if self.target is None or self.strategy is None:
            raise RuntimeError("Game not started. Call start_game().")

        if guess is None:
            guess = self.strategy.make_guess()
        else:
            if not isinstance(guess, int):
                raise ValueError("Guess must be an integer.")
            if not (self.config.min_range <= guess <= self.config.max_range):
                raise ValueError(
                    f"Guess must be between {self.config.min_range} and {self.config.max_range}"
                )

        self.guess_count += 1

        if guess == self.target:
            result = "correct"
        elif guess < self.target:
            result = "higher"
        else:
            result = "lower"

        self.strategy.update(guess, result)

        won = result == "correct"
        game_over = won or self.guess_count >= self.config.max_guesses

        return {
            "guess": guess,
            "result": result,
            "guess_count": self.guess_count,
            "won": won,
            "game_over": game_over,
            "target": self.target if game_over else None,
        }

    def auto_play_batch(self, count: int) -> List[Dict[str, Any]]:
        results = []
        for _ in range(count):
            self.start_game()
            while True:
                r = self.make_guess()
                if r["game_over"]:
                    results.append(
                        {
                            "won": r["won"],
                            "guesses": r["guess_count"],
                            "target": r["target"],
                        }
                    )
                    break
        return results


# ============================================================================
# SIMPLE CLI
# ============================================================================

class CLI:
    def __init__(self):
        self.config = Configuration()
        self.engine = GameEngine(self.config)

    def run(self):
        errors = self.config.validate()
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(" -", e)
            sys.exit(1)

        print("=" * 60)
        print("  NUMBER GUESSING SIMULATOR (LIGHT VERSION)")
        print("=" * 60)

        while True:
            print("\nMAIN MENU")
            print("1. Play single game")
            print("2. Auto-play multiple games")
            print("0. Exit")

            choice = input("Select option: ").strip()
            if choice == "1":
                self.play_single_game()
            elif choice == "2":
                self.auto_play_games()
            elif choice == "0":
                print("\nGoodbye!")
                break
            else:
                print("Invalid option.")

    def play_single_game(self):
        print("\nSINGLE GAME")
        mode = input("Mode (1=Manual, 2=Auto): ").strip()
        self.engine.start_game()
        print(
            f"Range: [{self.config.min_range}, {self.config.max_range}], "
            f"Strategy: {self.config.strategy}, "
            f"Max guesses: {self.config.max_guesses}"
        )

        while True:
            if mode == "1":
                try:
                    guess = int(input("Enter your guess: ").strip())
                except ValueError:
                    print("Please enter a valid integer.")
                    continue
            else:
                input("Press Enter for next auto guess...")
                guess = None

            try:
                r = self.engine.make_guess(guess)
            except Exception as e:
                print("Error:", e)
                continue

            print(f"Guess #{r['guess_count']}: {r['guess']}")

            if r["result"] == "correct":
                print("✅ Correct!")
            elif r["result"] == "higher":
                print("↑ Higher!")
            else:
                print("↓ Lower!")

            if r["game_over"]:
                if not r["won"]:
                    print(f"❌ Game over! Target was {r['target']}")
                print(f"Total guesses: {r['guess_count']}")
                break

    def auto_play_games(self):
        print("\nAUTO-PLAY")
        try:
            count = int(input("Number of games: ").strip())
        except ValueError:
            print("Invalid number.")
            return
        if count < 1 or count > 10000:
            print("Count must be between 1 and 10000.")
            return

        print(f"Running {count} games with strategy '{self.config.strategy}'...")
        start = time.time()
        results = self.engine.auto_play_batch(count)
        elapsed = time.time() - start

        wins = sum(1 for r in results if r["won"])
        total_guesses = sum(r["guesses"] for r in results)

        print("\nRESULTS")
        print(f"Games: {count}")
        print(f"Wins: {wins} ({wins / count * 100:.1f}%)")
        print(f"Losses: {count - wins} ({(count - wins) / count * 100:.1f}%)")
        print(f"Average guesses: {total_guesses / count:.2f}")
        print(f"Time: {elapsed:.2f}s, Games/sec: {count / elapsed:.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Lightweight guessing simulator")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in StrategyType],
        default="hybrid",
        help="Guessing strategy to use",
    )
    parser.add_argument("--min", dest="min_range", type=int, default=1)
    parser.add_argument("--max", dest="max_range", type=int, default=100)
    parser.add_argument("--max-guesses", dest="max_guesses", type=int, default=10)
    args = parser.parse_args()

    cli = CLI()
    cli.config.strategy = args.strategy
    cli.config.min_range = args.min_range
    cli.config.max_range = args.max_range
    cli.config.max_guesses = args.max_guesses
    cli.engine.config = cli.config

    cli.run()


if __name__ == "__main__":
    main()

