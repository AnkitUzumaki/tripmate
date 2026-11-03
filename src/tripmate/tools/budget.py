from typing import Literal

from tripmate.tools.registry import tool


@tool

def estimate_trip_budget(destination: str, days: int, travelers: int, tier: str)-> dict:
    rate = {
        "budget": {
            "hotel": 400,
            "transport": 100,
            "food": 100,
            "activity": 100,
        },
        "mid": {
            "hotel": 800,
            "transport": 200,
            "food": 200,
            "activity": 200,
        },
        "luxury":{
            "hotel": 1200,
            "transport": 300,
            "food": 300,
            "activity": 300,
        }
    }

    if days <=0 or travelers <=0:
        raise ValueError("Days and travelers must be greater than 0")
    if tier not in rate:
        raise ValueError("tier must be in budget, mid, or luxury")

    cost = rate[tier]

    per_person = (cost["hotel"] + cost["transport"] + cost["food"] + cost["activity"])


    total = per_person * travelers * days

    return {"destination": destination, "days": days, "travelers": travelers, "tier": tier, "per_person":per_person_per_day * days, "total":total}