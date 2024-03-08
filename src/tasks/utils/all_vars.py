# all possible variables used to fill specs
ALL_VARS = {
    # Combat
    "combat_biomes": ["forest", "plains", "extreme_hills"],
    "regular_biomes_mob": [
        "cow",
        "pig",
        "sheep",
        "chicken",
    ],
    "regular_biomes_night_mob": [
        "zombie",
        "spider",
        "skeleton",
        "creeper",
        "witch",
        "enderman",
    ],
    "end_mob": ["shulker", "endermite", "enderman",],
    "nether_mob": [
        "blaze",
        "ghast",
        "wither_skeleton",
        "zombie_pigman",
    ],
    "plains_mob": ["horse", "donkey"],
    "weapon_material": ["wooden", "iron", "diamond"],
    "armor_material": ["leather", "iron", "diamond"],
    # Harvest
    "quantity": [1, 8],
    ## wool and milk
    "cow_biomes": ["plains", "extreme_hills", "forest"],
    "sheep_biomes": ["plains", "extreme_hills", "forest"],
    ## mine
    "ore_type": ["iron_ore", "gold_ore", "diamond", "redstone", "coal", "cobblestone"],
    "pickaxe_material": ["wooden", "stone", "iron", "golden", "diamond"],
    ## most supported items (default only)
    "natural_items": [
        "nether_star",
        "blaze_rod",
        "ghast_tear",
        "nether_wart",
        "netherrack",
        "soul_sand",
        "chorus_flower",
        "chorus_fruit",
        "chorus_plant",
        "elytra",
        "end_stone",
        "ender_pearl",
        "apple",
        "beef",
        "beetroot",
        "beetroot_seeds",
        "bone",
        "brown_mushroom",
        "cactus",
        "carrot",
        "chicken",
        "dirt",
        "egg",
        "feather",
        "fish",
        "grass",
        "leaves",
        "log",
        "monster_egg",
        "mutton",
        "porkchop",
        "potato",
        "prismarine_shard",
        "pumpkin",
        "rabbit",
        "red_mushroom",
        "reeds",
        "sapling",
        "skull",
        "snowball",
        "spawn_egg",
        "sponge",
        "string",
        "totem_of_undying",
        "vine",
        "web",
        "wheat_seeds",
        "wheat",
    ],
    "craft_items": [
        "book",
        "carrot_on_a_stick",
        "clay",
        "crafting_table",
        "dye",
        "end_bricks",
        "end_rod",
        "ender_eye",
        "flint_and_steel",
        "glowstone",
        "gold_nugget",
        "iron_nugget",
        "iron_trapdoor",
        "lever",
        "nether_brick",
        "planks",
        "pumpkin_seeds",
        "red_nether_brick",
        "sandstone",
        "shears",
        "slime_ball",
        "stick",
        "stone_button",
        "stonebrick",
        "sugar",
        "torch",
        "trapped_chest",
        "wooden_button",
        "wool",
        "stone_pressure_plate",
    ],
    "crafting_table_items": [
        "anvil",
        "arrow",
        "banner",
        "beacon",
        "bed",
        "beetroot_soup",
        "boat",
        "bookshelf",
        "bowl",
        "bread",
        "bucket",
        "cake",
        "cauldron",
        "chest",
        "cookie",
        "end_crystal",
        "ender_chest",
        "fence",
        "fence_gate",
        "fire_charge",
        "fishing_rod",
        "flower_pot",
        "furnace",
        "glass_bottle",
        "glass_pane",
        "golden_apple",
        "hopper",
        "iron_bars",
        "ladder",
        "lead",
        "map",
        "minecart",
        "mushroom_stew",
        "painting",
        "paper",
        "pumpkin_pie",
        "rabbit_stew",
        "rail",
        "sea_lantern",
        "shield",
        "sign",
        "speckled_melon",
        "stone_slab",
        "trapdoor",
        "tripwire_hook",
        "wooden_door",
        "writable_book",
    ],
    "furnace_items": [
        "baked_potato",
        "brick",
        "cooked_beef",
        "cooked_chicken",
        "cooked_fish",
        "cooked_mutton",
        "cooked_porkchop",
        "cooked_rabbit",
        "glass",
        "gold_ingot",
        "iron_ingot",
        "quartz",
        "stone",
        "emerald",
        "netherbrick",
    ],
    ## core items
    ### most of the items here need trees in the biomes
    "biome_subset": ["plains", "jungle", "taiga", "forest", "swampland"],
    "natural_core": [
        "apple",
        "beef",
        "bone",
        "chicken",
        "log",
        "reeds",
        "web",
        "wheat",
    ],
    "hand_craft_core": [
        "flint_and_steel",
        "crafting_table",
        "planks",
        "shears",
        "stick",
        "sugar",
        "torch",
    ],
    "crafting_table_core": [
        "arrow",
        "chest",
        "shield",
        "fishing_rod",
        "bucket",
        "furnace",
    ],
    "furnace_core": [
        "cooked_beef",
        "glass",
        "gold_ingot",
        "iron_ingot",
        "brick",
        "stone",
    ],
    # Tech-tree
    "from_barehand_tools": ["wooden", "stone"],
    "from_barehand_tools_armor": ["iron", "golden", "diamond"],
    "from_wood_tools": ["stone"],
    "from_wood_tools_armor": ["iron", "golden", "diamond"],
    "from_stone_tools_armor": [
        "iron",
        "golden",
        "diamond",
    ],
    "from_iron_tools_armor": [
        "golden",
        "diamond",
    ],
    "from_gold_tools_armor": [
        "diamond",
    ],
    "target_tools": [
        "sword",
        "pickaxe",
        "axe",
        "hoe",
        "shovel",
    ],
    "target_armor": [
        "boots",
        "chestplate",
        "helmet",
        "leggings",
    ],
    "target_tools_armor": [
        "sword",
        "pickaxe",
        "axe",
        "hoe",
        "shovel",
        "boots",
        "chestplate",
        "helmet",
        "leggings",
    ],
    "redstone_list": [
        "redstone_block",
        "clock",
        "compass",
        "dispenser",
        "dropper",
        "observer",
        "piston",
        "redstone_lamp",
        "redstone_torch",
        "repeater",
        "detector_rail",
        "comparator",
        "activator_rail",
    ],
}