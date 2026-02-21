global shared_data

def is_prime(n):
	if n <= 1:
		return False
	if n <= 3:
		return True
	if n % 2 == 0 or n % 3 == 0:
		return False
	i = 5
	while i * i <= n:
		if n % i == 0 or n % (i + 2) == 0:
			return False
		i += 6
	return True

def generate_primes(limit=10000000):
	shared_data["primes"] = []
	num = 2
	while len(shared_data["primes"]) < limit:
		if is_prime(num):
			shared_data["primes"].append(num)
		num += 1
