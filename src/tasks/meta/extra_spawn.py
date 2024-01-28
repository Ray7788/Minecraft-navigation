from .utils import (
    check_below_height_condition,
    check_above_height_condition,
    check_above_light_level_condition,
    check_below_light_level_condition,
)


Target2SpawnItem = {
    # ore
    "cobblestone": "stone",
    "diamond": "diamond_ore",
    "redstone": "redstone_ore",
    "coal": "coal_ore",
    # ore byproduct
    "iron_ingot": "iron_ore",
    "gold_ingot": "gold_ore",
    # mob byproduct
    "leather": "cow",
    "mutton": "sheep",
    "porkchop": "pig",
    "beef": "cow",
    "bone": "skeleton",
    "web": "spider",
    "feather": "chicken",
    "wool": "sheep",
    "milk_bucket": "cow",
    # additional mob byproduct
    "rabbit_hide": "rabbit",
    "rabbit_foot": "rabbit",
    "rabbit": "rabbit",
    "string": "spider",
    "gunpowder": "creeper",
    "slime_ball": "slime",
    "ender_eye": "enderman",
    "blaze_rod": "blaze",
    "ghast_tear": "ghast",
    "zombie_head": "zombie",
    "blaze_rod": "blaze",
    "ghast_tear": "ghast",
    "slime_ball": "slime",
    # plant byproduct
    "wheat_seeds": "wheat",
    "beetroot_seeds": "beetroot",
    "pumpkin_seeds": "pumpkin",
    # additional supplement
    "melon_seeds": "melon",
    "carrot": "carrot",
    "potato": "potato",
    "sugar": "reeds",
    "egg": "chicken",
    # crafted items
    "diamond_pickaxe": "diamond",
    "diamond_sword": "diamond",
    "diamond_axe": "diamond",
    "diamond_shovel": "diamond",
    "diamond_hoe": "diamond",
    "iron_pickaxe": "iron_ingot",
    "iron_sword": "iron_ingot",
    "iron_axe": "iron_ingot",
    "iron_shovel": "iron_ingot",
    "iron_hoe": "iron_ingot",
    "gold_pickaxe": "gold_ingot",
    "gold_sword": "gold_ingot",
    "gold_axe": "gold_ingot",
    "gold_shovel": "gold_ingot",
    "gold_hoe": "gold_ingot",
    # existing mappings...
    "emerald": "emerald_ore",
    "lapis_lazuli": "lapis_lazuli_ore",
    "nether_quartz": "nether_quartz_ore",
    "obsidian": "lava_bucket",
    "flint": "gravel",
    # plant byproduct
    "apple": "oak_leaves",
    "cocoa_beans": "jungle_log",
    "bamboo": "bamboo_sapling",
    "sweet_berries": "sweet_berry_bush",
    "kelp": "kelp_plant",
    # crafted items
    "bread": "wheat",
    "cake": "milk_bucket",
    "cookie": "wheat",
    "pumpkin_pie": "pumpkin",
    "clock": "gold_ingot",
    "compass": "iron_ingot",
    "cookie": "wheat",
    "cake": "milk_bucket",
    "pumpkin_pie": "pumpkin",
    "tropical_fish_bucket": "tropical_fish",
    "pufferfish_bucket": "pufferfish",
    "cod_bucket": "cod",
    "salmon_bucket": "salmon",
    "axolotl_bucket": "axolotl",
    "glow_berries": "glow_berries",
    "powder_snow_bucket": "powder_snow",
    "bucket_of_tropical_fish": "tropical_fish",
    "bucket_of_pufferfish": "pufferfish",
    "bucket_of_cod": "cod",
    "bucket_of_salmon": "salmon",
    "bucket_of_axolotl": "axolotl",
}

# this dict specifies spawn conditions
SpawnItem2Condition = {
    # ore
    "stone": check_below_height_condition(height_threshold=60),
    "coal_ore": check_below_height_condition(height_threshold=60),
    "iron_ore": check_below_height_condition(height_threshold=50),
    "gold_ore": check_below_height_condition(height_threshold=29),
    "diamond_ore": check_below_height_condition(height_threshold=14),
    "redstone_ore": check_below_height_condition(height_threshold=16),
    # natural items
    "pumpkin": check_above_height_condition(height_threshold=62),
    "reeds": check_above_height_condition(height_threshold=62),
    "wheat": check_above_height_condition(height_threshold=62),
    "beetroot": check_above_height_condition(height_threshold=62),
    "potato": check_above_height_condition(height_threshold=62),
    "carrot": check_above_height_condition(height_threshold=62),
    # night mobs
    "zombie": check_below_light_level_condition(light_level_threshold=7),
    "creeper": check_below_light_level_condition(light_level_threshold=7),
    "spider": check_below_light_level_condition(light_level_threshold=7),
    "skeleton": check_below_light_level_condition(light_level_threshold=7),
    "witch": check_below_light_level_condition(light_level_threshold=7),
    "enderman": check_below_light_level_condition(light_level_threshold=7),
    "bat": check_below_light_level_condition(light_level_threshold=7),
    "husk": check_below_light_level_condition(light_level_threshold=7),
    "slime": check_below_light_level_condition(light_level_threshold=7),
    # day mobs
    "pig": check_above_light_level_condition(light_level_threshold=9),
    "cow": check_above_light_level_condition(light_level_threshold=9),
    "chicken": check_above_light_level_condition(light_level_threshold=9),
    "sheep": check_above_light_level_condition(light_level_threshold=9),
    "rabbit": check_above_light_level_condition(light_level_threshold=9),
    "horse": check_above_light_level_condition(light_level_threshold=9),
    "donkey": check_above_light_level_condition(light_level_threshold=9),
    "mooshroom": check_above_light_level_condition(light_level_threshold=9),
    "llama": check_above_light_level_condition(light_level_threshold=7),
}
