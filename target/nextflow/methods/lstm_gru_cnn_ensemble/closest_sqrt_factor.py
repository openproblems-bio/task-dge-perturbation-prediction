import math

def closest_sqrt_factor(x):
    """
    Finds the factor of x that is closest to the square root of x.

    Args:
        x: The number to find the closest factor of.

    Returns:
        The closest factor to the square root of x, or -1 if x is less than 1.
    """
    
    # Base cases
    if x < 1:
        return -1
    if x == 1:
        return 1
    
    # Start from the square root of x (rounded down)
    start = math.isqrt(x)

    # Check if the start value is a factor
    if x % start == 0:
        return start

    # Look for factors above and below the start
    for factor in range(start, 0, -1):
        if x % factor == 0:
            other_factor = x // factor
            return min(factor, other_factor, key=lambda f: abs(f - math.sqrt(x)))
