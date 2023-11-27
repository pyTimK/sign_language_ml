import os
import numpy as np
from .configuration import Configuration

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")


# Actions that we try to detect
class ACTIONS:
    Beginner = np.array(
        [
            "0_none",
            "1_yellow",
            "2_continue",
            "3_green",
            "4_black",
            "5_gray",
            "6_library",
            "7_cr",
            "8_church",
            "9_hospital",
            "10_home",
            "11_tomorrow",
            "12_feelings",
            "13_sad",
            "14_happy",
            "15_sick",
            "16_sister",
            "17_relatives",
            "18_grandpa",
            "19_father",
            "20_today",
        ]
    )

    Intermediate = np.array(
        [
            "0_none",
            "1_my_favorite_color_is_green",
            "2_what_color_do_you_want",
            "3_whats_the_color_of_the_shoes",
            "4_whats_the_color_of_the_thsirt",
            "5_are_you_okay",
            "6_im_excited",
            "7_what_happened",
            "8_where_does_it_hurt",
            "9_why_are_you_sad",
            "10_are_your_parents_strict",
            "11_call_your_sister_now",
            "12_how_are_your_parents",
            "13_my_family_consists_of_six",
            "14_where_is_your_brother",
            "15_can_you_help_me",
            "16_nice_to_meet_you",
            "17_see_you_later",
            "18_what_time_is_it",
            "19_ill_make_breakfast",
            "20_where_are_you_going",
        ]
    )


# Level Type
# Create a type Level with 'beginner' and 'intermediate' as its members


BEGINNER_CONFIG = Configuration(
    "BEGINNER",
    ACTIONS.Beginner,
    video_length=50,
    frame_length=30,
    start_folder=1,
)


INTERMEDIATE_CONFIG = Configuration(
    "INTERMEDIATE",
    ACTIONS.Intermediate,
    video_length=50,
    frame_length=40,
    start_folder=1,
)
