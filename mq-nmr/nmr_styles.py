class ColorManager:
    def __init__(self):
        self.color_dict = {
            "Rich Black": "#001219",
            "Blue Sapphire": "#005f73",
            "Viridian Green": "#0a9396",
            "Middle Blue Green": "#94d2bd",
            "Medium Champagne": "#e9d8a6",
            "Gamboge": "#ee9b00",
            "Alloy Orange": "#ca6702",
            "Rust": "#bb3e03",
            "Rufous": "#ae2012",
            "Ruby Red": "#9b2226"
        }

        self.color_dict_png = {
            "Dark Blue": "#161E49",
            "Medium Blue": "#295C77",
            "Light Blue": "#45B9BC",
            "Salmon Red": "#F66A49"
        }

    def get_color(self, name):
        return self.color_dict.get(name)

    def get_color_png(self, name):
        return self.color_dict_png.get(name)


class TerminalStyleManager:
    def __init__(self):
        self.styled_unicode_dict = {
            "success": "\033[1;38;5;82m\u2714\033[0m",   # Light neon green checkmark
            "failure": "\033[1;38;5;196m\u2717\033[0m",  # Light neon red cross
            "process": "\033[1;38;5;87m\u25B6\uFE0E\033[0m"  # Light neon yellow arrow
        }

    def get_style(self, name):
        return self.styled_unicode_dict.get(name)


"""

Example usage: 

color_manager = ColorManager()
style_manager = StyleManager()

print(color_manager.get_color("Rich Black"))
print(style_manager.get_style("success"))

"""
