import os
import numpy as np
from .configuration import Configuration

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")

# Actions that we try to detect
# TIMEOUT_SECONDS = 20
TIMEOUT_SECONDS = 40
THRESHOLD_NUM_MIN = 4


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
    Intermediate_1 = np.array(
        [
            "0_none",
            "1_my_favorite_color_is_green",
            "5_are_you_okay",
            "6_im_excited",
            "7_what_happened",
            "9_why_are_you_sad",
        ]
    )
    Intermediate_2 = np.array(
        [
            "0_none",
            "2_what_color_do_you_want",
            "8_where_does_it_hurt",
            "10_are_your_parents_strict",
            "11_call_your_sister_now",
            "15_can_you_help_me",
        ]
    )
    Intermediate_3 = np.array(
        [
            "0_none",
            "3_whats_the_color_of_the_shoes",
            "12_how_are_your_parents",
            "16_nice_to_meet_you",
            "17_see_you_later",
            "19_ill_make_breakfast",
        ]
    )

    Intermediate_4 = np.array(
        [
            "0_none",
            "4_whats_the_color_of_the_thsirt",
            "13_my_family_consists_of_six",
            "14_where_is_your_brother",
            "18_what_time_is_it",
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


INTERMEDIATE_1_CONFIG = Configuration(
    "INTERMEDIATE_1",
    ACTIONS.Intermediate_1,
    video_length=50,
    frame_length=40,
    start_folder=1,
)

INTERMEDIATE_2_CONFIG = Configuration(
    "INTERMEDIATE_2",
    ACTIONS.Intermediate_2,
    video_length=50,
    frame_length=40,
    start_folder=1,
)

INTERMEDIATE_3_CONFIG = Configuration(
    "INTERMEDIATE_3",
    ACTIONS.Intermediate_3,
    video_length=50,
    frame_length=40,
    start_folder=1,
)

INTERMEDIATE_4_CONFIG = Configuration(
    "INTERMEDIATE_4",
    ACTIONS.Intermediate_4,
    video_length=50,
    frame_length=40,
    start_folder=1,
)


filename_to_phrase = {
    "0_none": "None",
    "1_yellow": "Yellow",
    "2_continue": "Continue",
    "3_green": "Green",
    "4_black": "Black",
    "5_gray": "Gray",
    "6_library": "Library",
    "7_cr": "CR",
    "8_church": "Church",
    "9_hospital": "Hospital",
    "10_home": "Home",
    "11_tomorrow": "Tomorrow",
    "12_feelings": "Feelings",
    "13_sad": "Sad",
    "14_happy": "Happy",
    "15_sick": "Sick",
    "16_sister": "Sister",
    "17_relatives": "Relatives",
    "18_grandpa": "Grandpa",
    "19_father": "Father",
    "20_today": "Today",
    "1_my_favorite_color_is_green": "My favorite color is green",
    "5_are_you_okay": "Are you okay?",
    "6_im_excited": "I'm excited",
    "7_what_happened": "What happened?",
    "9_why_are_you_sad": "Why are you sad?",
    "2_what_color_do_you_want": "What color do you want?",
    "8_where_does_it_hurt": "Where does it hurt?",
    "10_are_your_parents_strict": "Are your parents strict?",
    "11_call_your_sister_now": "Call your sister now",
    "15_can_you_help_me": "Can you help me?",
    "3_whats_the_color_of_the_shoes": "What's the color of the shoes?",
    "12_how_are_your_parents": "How are your parents?",
    "16_nice_to_meet_you": "Nice to meet you",
    "17_see_you_later": "See you later",
    "19_ill_make_breakfast": "I'll make breakfast",
    "4_whats_the_color_of_the_thsirt": "What's the color of the thsirt?",
    "13_my_family_consists_of_six": "My family consists of six",
    "14_where_is_your_brother": "Where is your brother?",
    "18_what_time_is_it": "What time is it?",
    "20_where_are_you_going": "Where are you going?",
}
