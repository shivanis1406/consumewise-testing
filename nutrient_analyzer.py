# Nutrient thresholds for solids and liquids
thresholds = {
    'solid': {
        'calories': 250,
        'sugar': 3,
        'fat': 4.2,
        'salt': 625
    },
    'liquid': {
        'calories': 70,
        'sugar': 2,
        'fat': None,  # Not applicable
        'salt': 175
    }
}

# Function to calculate percentage difference from threshold
def calculate_percentage_difference(value, threshold):
    if threshold is None:
        return None  # For nutrients without a threshold
    return ((value - threshold) / threshold) * 100

# Function to analyze nutrients and calculate differences
def analyze_nutrients(product_type, calories, sugar, salt, serving_size, added_fat=None):
    threshold_data = thresholds.get(product_type)
    if not threshold_data:
        raise ValueError(f"Invalid product type: {product_type}")

    scaled_calories = (calories / serving_size) * 100
    scaled_sugar = (sugar / serving_size) * 100
    scaled_salt = (salt / serving_size) * 100
    scaled_fat = (added_fat / serving_size) * 100 if added_fat is not None else None

    return {
        'calories': {
            'value': scaled_calories,
            'threshold': threshold_data['calories'],
            'difference': scaled_calories - threshold_data['calories'],
            'percentageDiff': calculate_percentage_difference(scaled_calories, threshold_data['calories'])
        },
        'sugar': {
            'value': scaled_sugar,
            'threshold': threshold_data['sugar'],
            'difference': scaled_sugar - threshold_data['sugar'],
            'percentageDiff': calculate_percentage_difference(scaled_sugar, threshold_data['sugar'])
        },
        'salt': {
            'value': scaled_salt,
            'threshold': threshold_data['salt'],
            'difference': scaled_salt - threshold_data['salt'],
            'percentageDiff': calculate_percentage_difference(scaled_salt, threshold_data['salt'])
        },
        'fat': {
            'value': scaled_fat,
            'threshold': threshold_data['fat'],
            'difference': scaled_fat - threshold_data['fat'] if scaled_fat is not None else None,
            'percentageDiff': calculate_percentage_difference(scaled_fat, threshold_data['fat']) if scaled_fat is not None else None
        }
    }

# Example of how these functions can be called in the main code
#if __name__ == "__main__":
#    product_type = 'solid'  # 'solid' or 'liquid'
#    calories = 300
#    sugar = 4
#    salt = 700
#    serving_size = 100
#    added_fat = 5  # Optional

#    result = analyze_nutrients(product_type, calories, sugar, salt, serving_size, added_fat)
#    print(result)
