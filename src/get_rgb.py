# RETURN RGB VALUE FOR THE GIVEN NAME (TEXT)
#   Reference:
#       [1] https://www.rapidtables.com/web/color/RGB_Color.html
#       [2] https://www.schemecolor.com/


def get_rgb(color_name):
    rgb_value = []
    if color_name == "White" or color_name == "white" or color_name == "WHITE":
        rgb_value = (255, 255, 255)
    elif color_name == "Black" or color_name == "black" or color_name == "BLACK":
        rgb_value = (0, 0, 0)
    elif color_name == "Red" or color_name == "red" or color_name == "RED":
        rgb_value = (255, 0, 0)
    elif color_name == "Blue" or color_name == "blue" or color_name == "BLUE":
        rgb_value = (0, 0, 255)
    elif color_name == "Yellow" or color_name == "yellow" or color_name == "YELLOW":
        rgb_value = (255, 255, 0)
    elif color_name == "Cyan" or color_name == "cyan" or color_name == "CYAN":
        rgb_value = (0, 255, 255)
    elif color_name == "Magenta" or color_name == "magenta" or color_name == "MAGENTA":
        rgb_value = (255, 0, 255)
    elif color_name == "Silver" or color_name == "silver" or color_name == "SILVER":
        rgb_value = (192, 192, 192)
    elif color_name == "Gray" or color_name == "gray" or color_name == "GRAY":
        rgb_value = (128, 128, 128)
    elif color_name == "Green" or color_name == "green" or color_name == "GREEN":
        rgb_value = (0, 255, 0)
    elif color_name == "Purple" or color_name == "purple" or color_name == "PURPLE":
        rgb_value = (128, 0, 128)

    # RED ~ ORANGE
    elif color_name == "Maroon" or color_name == "maroon" or color_name == "MAROON":
        rgb_value = (128, 0, 0)
    elif color_name == "Dark Red" or color_name == "dark red" or color_name == "DARK RED":
        rgb_value = (139, 0, 0)
    elif color_name == "Brown" or color_name == "brown" or color_name == "BROWN":
        rgb_value = (165, 42, 42)
    elif color_name == "Firebrick" or color_name == "firebrick" or color_name == "FIREBRICK":
        rgb_value = (178, 34, 34)
    elif color_name == "Crimson" or color_name == "crimson" or color_name == "CRIMSON":
        rgb_value = (220, 20, 60)
    elif color_name == "Tomato" or color_name == "tomato" or color_name == "TOMATO":
        rgb_value = (255, 99, 71)
    elif color_name == "Coral" or color_name == "coral" or color_name == "CORAL":
        rgb_value = (255, 127, 80)
    elif color_name == "Indian Red" or color_name == "indian red" or color_name == "INDIAN RED":
        rgb_value = (255, 127, 80)
    elif color_name == "Light Coral" or color_name == "light coral" or color_name == "LIGHT CORAL":
        rgb_value = (240, 128, 128)
    elif color_name == "Dark Salmon" or color_name == "dark salmon" or color_name == "DARK SALMON":
        rgb_value = (233, 150, 122)
    elif color_name == "Salmon" or color_name == "salmon" or color_name == "SALMON":
        rgb_value = (250, 128, 114)
    elif color_name == "Light Salmon" or color_name == "light salmon" or color_name == "LIGHT SALMON":
        rgb_value = (255, 160, 122)
    elif color_name == "Orange Red" or color_name == "orange red" or color_name == "ORANGE RED":
        rgb_value = (255, 69, 0)
    elif color_name == "Dark Orange" or color_name == "dark orange" or color_name == "DARK ORANGE":
        rgb_value = (255, 140, 0)
    elif color_name == "Orange" or color_name == "orange" or color_name == "ORANGE":
        rgb_value = (255, 165, 0)

    # YELLOW ~ GREEN
    elif color_name == "Gold" or color_name == "gold" or color_name == "GOLD":
        rgb_value = (255, 215, 0)
    elif color_name == "Dark Golden Rod" or color_name == "dark golden rod" or color_name == "DARK GOLDEN ROD":
        rgb_value = (184, 134, 11)
    elif color_name == "Golden Rod" or color_name == "golden rod" or color_name == "GOLDEN ROD":
        rgb_value = (218, 165, 32)
    elif color_name == "Pale Golden Rod" or color_name == "pale golden rod" or color_name == "PALE GOLDEN ROD":
        rgb_value = (238, 232, 170)
    elif color_name == "Dark Khaki" or color_name == "dark khaki" or color_name == "DARK KHAKI":
        rgb_value = (189, 183, 107)
    elif color_name == "Khaki" or color_name == "khaki" or color_name == "KHAKI":
        rgb_value = (240, 230, 140)
    elif color_name == "Olive" or color_name == "olive" or color_name == "OLIVE":
        rgb_value = (128, 128, 0)
    elif color_name == "Yellow Green" or color_name == "yellow green" or color_name == "YELLOW GREEN":
        rgb_value = (154, 205, 50)
    elif color_name == "Dark Olive Green" or color_name == "dark olive green" or color_name == "DARK OLIVE GREEN":
        rgb_value = (85, 107, 47)
    elif color_name == "Olive Drab" or color_name == "olive drab" or color_name == "OLIVE DRAB":
        rgb_value = (107, 142, 35)
    elif color_name == "Lawn Green" or color_name == "lawn green" or color_name == "LAWN GREEN":
        rgb_value = (124, 252, 0)
    elif color_name == "Green Yellow" or color_name == "green yellow" or color_name == "GREEN YELLOW":
        rgb_value = (173, 255, 47)
    elif color_name == "Dark Green" or color_name == "dark green" or color_name == "DARK GREEN":
        rgb_value = (0, 100, 0)
    elif color_name == "Forest Green" or color_name == "forest green" or color_name == "FOREST GREEN":
        rgb_value = (34, 139, 34)
    elif color_name == "Lime" or color_name == "lime" or color_name == "LIME":
        rgb_value = (0, 255, 0)
    elif color_name == "Lime Green" or color_name == "lime green" or color_name == "LIME GREEN":
        rgb_value = (50, 205, 50)
    elif color_name == "Light Green" or color_name == "light green" or color_name == "LIGHT GREEN":
        rgb_value = (144, 238, 144)
    elif color_name == "Pale Green" or color_name == "pale green" or color_name == "PALE GREEN":
        rgb_value = (152, 251, 152)

    # BLUE
    elif color_name == "Dark Cyan" or color_name == "dark cyan" or color_name == "DARK CYAN":
        rgb_value = (0,139,139)
    elif color_name == "Dark Turquoise" or color_name == "dark turquoise" or color_name == "DARK TURQUOISE":
        rgb_value = (0, 206, 209)
    elif color_name == "Turquoise" or color_name == "turquoise" or color_name == "TURQUOISE":
        rgb_value = (64, 224, 208)
    elif color_name == "Steel Blue" or color_name == "steel blue" or color_name == "STEEL BLUE":
        rgb_value = (70, 130, 180)
    elif color_name == "Corn Flower Blue" or color_name == "corn flower blue" or color_name == "CORN FLOWER BLUE":
        rgb_value = (100, 149, 237)
    elif color_name == "Deep Sky Blue" or color_name == "deep sky blue" or color_name == "DEEP SKY BLUE":
        rgb_value = (0, 191, 255)
    elif color_name == "Dodger Blue" or color_name == "dodger blue" or color_name == "DODGER BLUE":
        rgb_value = (0, 191, 255)
    elif color_name == "Light Blue" or color_name == "light blue" or color_name == "LIGHT BLUE":
        rgb_value = (173, 216, 230)
    elif color_name == "Sky Blue" or color_name == "sky blue" or color_name == "SKY BLUE":
        rgb_value = (135, 206, 235)
    elif color_name == "Light Sky Blue" or color_name == "light sky blue" or color_name == "LIGHT SKY BLUE":
        rgb_value = (135, 206, 250)
    elif color_name == "Midnight Blue" or color_name == "midnight blue" or color_name == "MIDNIGHT BLUE":
        rgb_value = (25, 25, 112)
    elif color_name == "Navy" or color_name == "navy" or color_name == "NAVY":
        rgb_value = (0, 0, 128)
    elif color_name == "Dark Blue" or color_name == "dark blue" or color_name == "DARK BLUE":
        rgb_value = (0, 0, 139)
    elif color_name == "Medium Blue" or color_name == "medium blue" or color_name == "MEDIUM BLUE":
        rgb_value = (0, 0, 205)
    elif color_name == "Royal Blue" or color_name == "royal blue" or color_name == "ROYAL BLUE":
        rgb_value = (65, 105, 225)

    elif color_name == "Deep Pink" or color_name == "deep pink" or color_name == "DEEP PINK":
        rgb_value = (255, 20, 147)
    elif color_name == "Slate Gray" or color_name == "slate gray" or color_name == "SLATE GRAY":
        rgb_value = (112, 128, 144)
    elif color_name == "Dark Slate Blue" or color_name == "dark slate blue" or color_name == "DARK SLATE BLUE":
        rgb_value = (72, 61, 139)
    elif color_name == "Medium Slate Blue" or color_name == "medium slate blue" or color_name == "MEDIUM SLATE BLUE":
        rgb_value = (123, 104, 238)

    elif color_name == "Dark Violet" or color_name == "dark violet" or color_name == "DARK VIOLET":
        rgb_value = (148, 0, 211)
    elif color_name == "Medium Violet Red" or color_name == "medium violet red" or color_name == "MEDIUM VIOLET RED":
        rgb_value = (199, 21, 133)

    # BLACK
    elif color_name == "Dim Gray" or color_name == "dim gray" or color_name == "DIM GRAY":
        rgb_value = (105, 105, 105)
    elif color_name == "Dark Gray" or color_name == "dark gray" or color_name == "DARK GRAY":
        rgb_value = (169, 169, 169)
    elif color_name == "Very Dark Gray" or color_name == "very dark gray" or color_name == "VERY DARK GRAY":
        rgb_value = (72, 72, 72)
    elif color_name == "Silver" or color_name == "silver" or color_name == "SILVER":
        rgb_value = (192, 192, 192)
    elif color_name == "Light Gray" or color_name == "light gray" or color_name == "LIGHT GRAY":
        rgb_value = (211, 211, 211)
    elif color_name == "Gainsboro" or color_name == "gainsboro" or color_name == "GAINSBORO":
        rgb_value = (220, 220, 220)
    elif color_name == "White Smoke" or color_name == "white smoke" or color_name == "WHITE SMOKE":
        rgb_value = (245, 245, 245)

    # PASTEL: https://www.schemecolor.com/
    elif color_name == "Pastel Red" or color_name == "pastel red" or color_name == "PASTEL RED":
        rgb_value = (249, 102, 94)
    elif color_name == "Dark Pastel Red" or color_name == "dark pastel red" or color_name == "DARK PASTEL RED":
        rgb_value = (193, 59, 36)
    elif color_name == "Pastel Orange" or color_name == "pastel orange" or color_name == "PASTEL ORANGE":
        rgb_value = (255, 179, 70)
    elif color_name == "Pastel Yellow" or color_name == "pastel yellow" or color_name == "PASTEL YELLOW":
        rgb_value = (255, 244, 156)
    elif color_name == "Pastel Green" or color_name == "pastel green" or color_name == "PASTEL GREEN":
        rgb_value = (249, 255, 203)
    elif color_name == "Pastel Blue" or color_name == "pastel blue" or color_name == "PASTEL BLUE":
        rgb_value = (175, 199, 208)
    elif color_name == "Dark Pastel Blue" or color_name == "dark pastel blue" or color_name == "DARK PASTEL BLUE":
        rgb_value = (121, 159, 203)

    # DARK TO LIGHT BLUE GRADIENT: https://www.schemecolor.com/
    elif color_name == "Baby Blue Eyes" or color_name == "baby blue eyes" or color_name == "BABY BLUE EYES":
        rgb_value = (158, 194, 255)
    elif color_name == "Vista Blue" or color_name == "vista blue" or color_name == "VISTA BLUE":
        rgb_value = (123, 159, 242)
    elif color_name == "Han Blue" or color_name == "han blue" or color_name == "HAN BLUE":
        rgb_value = (66, 89, 195)
    elif color_name == "Egyptian Blue" or color_name == "egyptian blue" or color_name == "EGYPTIAN BLUE":
        rgb_value = (33, 42, 165)

    # HAPPY MONSOON: https://www.schemecolor.com/
    elif color_name == "Pigment" or color_name == "Pigment" or color_name == "PIGMENT":
        rgb_value = (237, 28, 34)
    elif color_name == "Metallic Yellow" or color_name == "metallic yellow" or color_name == "METALLIC YELLOW":
        rgb_value = (254, 201, 7)
    elif color_name == "Dark Lemon Lime" or color_name == "dark lemon lime" or color_name == "DARK LEMON LIME":
        rgb_value = (124, 187, 21)
    elif color_name == "Light Sky Blue" or color_name == "light sky blue" or color_name == "LIGHT SKY BLUE":
        rgb_value = (146, 208, 255)
    elif color_name == "Button Blue" or color_name == "button blue" or color_name == "BUTTON BLUE":
        rgb_value = (48, 173, 229)
    elif color_name == "Bright Navy Blue" or color_name == "bright navy blue" or color_name == "BRIGHT NAVY BLUE":
        rgb_value = (19, 115, 199)

    # PASTEL-OTHERS: https://www.schemecolor.com/
    elif color_name == "Coral Reef" or color_name == "coral reef" or color_name == "CORAL REEF":
        rgb_value = (255, 117, 109)
    elif color_name == "Crayola" or color_name == "crayola" or color_name == "CRAYOLA":
        rgb_value = (101, 212, 232)
    elif color_name == "Blueberry" or color_name == "blueberry" or color_name == "BLUEBERRY":
        rgb_value = (66, 133, 244)

    else:
        print("Do nothing.")

    rgb_value = (float(rgb_value[0] / 255.0), float(rgb_value[1] / 255.0), float(rgb_value[2] / 255.0))

    return rgb_value


def get_color_blurred(hcolor, alpha=1.2):
    r_new = max(min(hcolor[0] * alpha, 1.0), 0.0)
    g_new = max(min(hcolor[1] * alpha, 1.0), 0.0)
    b_new = max(min(hcolor[2] * alpha, 1.0), 0.0)
    hcolor_new = (r_new, g_new, b_new)

    return hcolor_new


def get_color_mix(alpha, hcolor1, hcolor2):
    """
    Get mix color.
    """
    new_hcolor_r = hcolor1[0] * (1 - alpha) + hcolor2[0] * alpha
    new_hcolor_g = hcolor1[1] * (1 - alpha) + hcolor2[1] * alpha
    new_hcolor_b = hcolor1[2] * (1 - alpha) + hcolor2[2] * alpha
    hcolor_new = (new_hcolor_r, new_hcolor_g, new_hcolor_b)

    return hcolor_new
