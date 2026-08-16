import shutil
import unittest
import uuid
from dataclasses import fields

from tetris_ai.config import AppConfig, CONFIG
from tetris_ai.engine import FitnessWeights, RuleWeights
from tetris_ai.paths import RUNTIME_PATHS, ensure_inside_project
from tetris_ai.settings import SETTING_DEFINITIONS, RuntimeSettings, SettingsManager, apply_runtime_settings, parse_runtime_settings, setting_values


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = ensure_inside_project(RUNTIME_PATHS["temp"] / f"settings-test-{uuid.uuid4().hex}")
        self.directory.mkdir(parents=True)
        self.manager = SettingsManager(self.directory / "settings.json")

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_vram_limit_round_trip(self):
        self.manager.save(RuntimeSettings(vram_limit_mib=8192, population_size=4200))
        loaded = self.manager.load()
        self.assertEqual(loaded.vram_limit_mib, 8192)
        self.assertEqual(loaded.population_size, 4200)
        self.assertFalse((self.directory / "settings.json.tmp").exists())

    def test_missing_and_invalid_settings_use_auto(self):
        self.assertEqual(self.manager.load().vram_limit_mib, 0)
        self.manager.path.write_text("invalid", encoding="utf-8")
        self.assertEqual(self.manager.load().vram_limit_mib, 0)
        self.assertEqual(self.manager.load().population_size, 0)

    def test_invalid_override_values_fall_back_without_breaking_startup(self):
        self.manager.path.write_text('{"vram_limit_mib": 4096, "overrides": {"mutation_rate": 5}}', encoding="utf-8")
        loaded = self.manager.load()
        self.assertEqual(loaded.vram_limit_mib, 4096)
        self.assertEqual(loaded.overrides, {})

    def test_old_settings_format_remains_compatible(self):
        self.manager.path.write_text('{"vram_limit_mib": 6144, "population_size": 777}', encoding="utf-8")
        loaded = self.manager.load()
        self.assertEqual(loaded.vram_limit_mib, 6144)
        self.assertEqual(loaded.population_size, 777)
        self.assertEqual(loaded.overrides, {})

    def test_old_single_agent_setting_scales_evolution_groups(self):
        applied = apply_runtime_settings(CONFIG, RuntimeSettings(population_size=1))
        self.assertEqual(applied.population_size, 1)
        self.assertEqual(applied.elite_count, 1)
        self.assertEqual(applied.parent_pool_size, 1)

    def test_schema_covers_every_editable_config_leaf(self):
        top_level = {item.name for item in fields(AppConfig)} - {"version", "output_size", "rule_weights", "fitness_weights"}
        nested = {f"rule.{item.name}" for item in fields(RuleWeights)}
        nested |= {f"fitness.{item.name}" for item in fields(FitnessWeights)}
        self.assertEqual({definition.key for definition in SETTING_DEFINITIONS}, top_level | nested)

    def test_all_values_parse_save_and_apply(self):
        values = setting_values(CONFIG)
        values["language"] = "en"
        values["board_width"] = "12"
        values["population_size"] = "2048"
        values["hidden_sizes"] = "128, 80, 32"
        values["mutation_scale"] = "0,2"
        values["rule.holes"] = "7.5"
        values["fitness.completed_lines"] = "1500"
        values["neural_network_vram_limit_mib"] = "9216"
        runtime = parse_runtime_settings(values)
        self.manager.save(runtime)
        applied = apply_runtime_settings(CONFIG, self.manager.load())
        self.assertEqual(applied.board_width, 12)
        self.assertEqual(applied.output_size, 48)
        self.assertEqual(applied.population_size, 2048)
        self.assertEqual(applied.hidden_sizes, (128, 80, 32))
        self.assertEqual(applied.mutation_scale, 0.2)
        self.assertEqual(applied.rule_weights.holes, 7.5)
        self.assertEqual(applied.fitness_weights.completed_lines, 1500.0)
        self.assertEqual(applied.neural_network_vram_limit_mib, 9216)
        self.assertEqual(applied.language, "en")

    def test_old_sensor_interval_is_migrated_to_one_and_half_seconds(self):
        self.manager.path.write_text('{"overrides": {"language": "en", "hardware_monitor_interval": 1.0}}', encoding="utf-8")
        loaded = self.manager.load()
        applied = apply_runtime_settings(CONFIG, loaded)
        self.assertEqual(applied.hardware_monitor_interval, 1.5)
        self.assertEqual(applied.language, "en")

    def test_dependency_validation_rejects_invalid_population_groups(self):
        values = setting_values(CONFIG)
        values["population_size"] = "10"
        values["elite_count"] = "11"
        with self.assertRaisesRegex(ValueError, "Elite agents"):
            parse_runtime_settings(values)
        values["elite_count"] = "5"
        values["parent_pool_size"] = "4"
        with self.assertRaisesRegex(ValueError, "Parent pool"):
            parse_runtime_settings(values)

    def test_numeric_validation_rejects_out_of_range_and_non_finite_values(self):
        values = setting_values(CONFIG)
        values["mutation_rate"] = "1.1"
        with self.assertRaisesRegex(ValueError, "Mutation probability"):
            parse_runtime_settings(values)
        values = setting_values(CONFIG)
        values["mutation_scale"] = "nan"
        with self.assertRaisesRegex(ValueError, "Invalid value"):
            parse_runtime_settings(values)


if __name__ == "__main__":
    unittest.main()
