from minerl.herobraine.hero.handlers.agent.actions.smelt import *

# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton
class SmeltAction(SmeltItemNearby):
    """
    Overload!!
    An action handler for crafting items when agent is in view of a crafting table

    Note when used along side Craft Item, block lists must be disjoint or from_universal will fire multiple times
    """
    def to_string(self):
        return 'nearbySmelt'