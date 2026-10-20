# run.py
import json
from workflow import run_workflow

test_user_1 = """
Restaurant Visit History:
- Visited "Green Bowl" (Vegan, $$) 8 times
- Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
- Visited "Juice Lab" (Smoothies, $) 3 times

Social Media Posts:
- "Loving my plant-based journey! 🌱"
- "This gluten-free Mediterranean bowl is amazing!"
- "Fresh juice is the best way to start the day."

Dietary Restrictions: Vegan, Gluten-Free
"""

if __name__ == "__main__":
    print("=" * 80)
    print("TEST CASE 1: Health-Conscious User")
    print("=" * 80)

    try:
        result_1 = run_workflow(test_user_1)
        print("\n" + "=" * 80)
        print("FINAL RECOMMENDATIONS")
        print("=" * 80)
        print(json.dumps(result_1["final_recommendations"], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nError: {e}")
        raise