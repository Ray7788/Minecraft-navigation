from minerl.herobraine.hero.handlers.agent.actions import CraftAction

class CraftWithTableAction(CraftAction):
    """
    An action handler for crafting items when agent is in view of a crafting table
    """

    _command = "craftNearby"

    def to_string(self):
        return "craft_with_table"

    def xml_template(self) -> str:
        return str("<NearbyCraftCommands/>")