---
title: update4 10 Jan 2024
date: 2024-01-11 12:17:14
tags:
---
Update
=======
Outcome
-----
* Completed essential transfer from Minedojo to MineRL.
  * Add more wrappers including operation logic of interaction for Minecraft Client, task setting...
  * still use Minedojo's defined tasks, task categories, task allocation
  * Add helper functions for env initialisation: inventory, block, player's slot(item, health, food), partial settings are based on different tasks, but now we could only set fixed env settings
  * Delete uncessary/repeated functions and make the project volume smaller.
* Exitsintg problem； lidar aprt(Minedojo). Needs to overwrite Malmo (lowest level: build.gradle)

TODO with problems
------
* Implement transfer learning plan：
  * existing problem: Train existing model with LoRA, using peft? `openai/clip-vit-base-patch16
` https://radekosmulski.com/how-to-fine-tune-a-tranformer-pt-2/ check: https://huggingface.co/openai/clip-vit-base-patch16
  * Prepare additional dataset? if so, should I use additional training dataset？
  * 