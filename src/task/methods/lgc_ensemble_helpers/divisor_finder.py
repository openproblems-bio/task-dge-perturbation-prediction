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

def find_balanced_divisors(n, threshold=100):
    """
    Finds a number greater than or equal to n that has two divisors with the smallest 
    difference within a specified threshold.

    Args:
        n: The starting number to find balanced divisors for.
        threshold: The maximum allowable difference between the two divisors. 
        Default is 100.

    Returns:
        A tuple containing the number with balanced divisors and a list of the two divisors.
    """

    # Start with the initial number n
    current_n = n
    while True:

        # Find the factor of current_n that is closest to its square root
        divisor1 = closest_sqrt_factor(current_n)

        # Calculate the corresponding divisor by dividing current_n by divisor1
        divisor2 = current_n//divisor1

        # Check if divisor2 is actually an integer
        # This is a safeguard to ensure divisor2 is a whole number
        if divisor2 != (current_n/divisor1):
            raise ValueError(f"divisor2 is not an integer: {divisor2}")
        
        # Check if the absolute difference between the two divisors is less than the threshold
        if abs(divisor1 - divisor2) < threshold:
            return current_n, [divisor1, divisor2]

        # If the difference is not acceptable, increment the number and try again
        current_n += 1
