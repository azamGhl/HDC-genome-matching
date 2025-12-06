import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


class BioHD:
    def __init__(self, hypervector_dimension, dna_sequence, dna_query_length):
        # Hyperparameters
        self.hypervector_dimension = hypervector_dimension
        self.dna_sequence = dna_sequence.upper()  # Ensure uppercase for consistency
        self.dna_query_length = dna_query_length
        self.base_hypervectors = self.create_base_hypervectors()

    def create_base_hypervectors(self):
        # Create binary hypervectors for each nucleotide
        nucleotides = {
            'A': np.random.randint(0, 2, size=self.hypervector_dimension),
            'C': np.random.randint(0, 2, size=self.hypervector_dimension),
            'G': np.random.randint(0, 2, size=self.hypervector_dimension),
            'T': np.random.randint(0, 2, size=self.hypervector_dimension)
        }
        return nucleotides  # Return the dictionary

    def permute(self, arr, k):
        # Circularly shift the array by k positions
        k = k % len(arr)
        left = arr[-k:]
        right = arr[:-k]
        return np.concatenate((left, right))

    def ProteinSequenceEncoding(self, n):
        """
        Splits the DNA sequence into chunks of length `dna_query_length`,
        then further splits each chunk into subwords of length `n`.
        """
        chunks = [
            self.dna_sequence[i:i + self.dna_query_length]
            for i in range(0, len(self.dna_sequence) - self.dna_query_length + 1)
        ]
        chunks_divide_n = []
        for word in chunks:
            subwords = []
            for i in range(0, len(word), n):
                subword = word[i:i + n]
                if len(subword) == n:
                    subwords.append(subword)
            if subwords:
                chunks_divide_n.append(subwords)
        return chunks_divide_n

    def bind(self, subsequence):
        bound_vector = np.zeros(self.hypervector_dimension, dtype=int)

        # Apply permutation and XOR for each nucleotide in the subsequence
        for idx, nucleotide in enumerate(subsequence):
            if nucleotide not in self.base_hypervectors:
                raise ValueError(f"Invalid nucleotide '{nucleotide}' encountered.")
            permuted_vector = self.permute(self.base_hypervectors[nucleotide],
                                           idx)  # Apply permutation based on position
            bound_vector = np.bitwise_xor(bound_vector, permuted_vector)  # Bind using XOR

        return bound_vector

    def generate_hypervectors(self, chunks_divide_n):
        hypervectors = []
        for subword_group in chunks_divide_n:
            group_hypervectors = []
            for subword in subword_group:
                if len(subword) == 0:
                    continue  # Skip empty subwords
                bound_hv = self.bind(subword)
                group_hypervectors.append(bound_hv)
            hypervectors.append(group_hypervectors)
        return hypervectors

    def bundle_vectors(self, vectors):
        if len(vectors) == 0:
            return np.zeros(self.hypervector_dimension, dtype=int)
        # Compute the elementwise sum of the input vectors
        sum_vector = np.sum(vectors, axis=0)

        # Bundle the vectors by taking the elementwise majority
        R = np.where(sum_vector >= len(vectors) / 2, 1, 0)

        return R

    def create_encoded_library(self, subsequence_groups):
        """
        Bundles hypervectors for each group of subsequences to create the encoded library.
        """
        bundled_hypervectors = []  # List to store the bundled hypervectors

        # Generate hypervectors for all subsequence groups
        genom_vectors = self.generate_hypervectors(subsequence_groups)

        gen_len = len(genom_vectors)  # Length of the generated hypervectors

        # For each group of hypervectors, bundle them
        for i in range(gen_len):
            R = self.bundle_vectors(genom_vectors[i])  # Bundle vectors for group i
            bundled_hypervectors.append(R)  # Append bundled hypervector to list

        return bundled_hypervectors

    def create_test_data(self, num_exact, num_approx, num_mutations=1):
        """
        Create two separate test datasets for exact matches and approximate matches.

        Returns two test sets:
        - exact_test_data: Exact matches and non-matches.
        - approx_test_data: Approximate matches and non-matches.
        """
        exact_matches = []
        approximate_matches = []
        non_matches = []

        sequence_length = self.dna_query_length
        dna_length = len(self.dna_sequence)

        # Precompute all possible exact matches for efficiency
        possible_matches = set()
        for i in range(dna_length - sequence_length + 1):
            substring = self.dna_sequence[i:i + sequence_length]
            possible_matches.add(substring)

        # Convert to list for random sampling
        possible_matches = list(possible_matches)

        # Check if enough exact matches are available
        if len(possible_matches) < num_exact:
            raise ValueError(f"Not enough unique substrings in dna_sequence to generate {num_exact} exact matches.")

        # Randomly select exact matches
        selected_exact = random.sample(possible_matches, num_exact)
        exact_matches = [(seq, True) for seq in selected_exact]

        # Generate approximate matches by mutating exact matches
        for exact_seq in selected_exact[:num_approx]:
            approx_seq = self.mutate_sequence(exact_seq, num_mutations)
            approximate_matches.append((approx_seq, True))

        # Function to generate a random DNA sequence
        def generate_random_dna(length):
            return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

        # Generate non-matching sequences
        non_matches_set = set()
        attempts = 0
        max_attempts = num_exact * 10  # To prevent infinite loop

        while len(non_matches_set) < num_exact and attempts < max_attempts:
            rand_seq = generate_random_dna(sequence_length)
            if rand_seq not in possible_matches and rand_seq not in [seq for seq, _ in approximate_matches]:
                non_matches_set.add(rand_seq)
            attempts += 1

        if len(non_matches_set) < num_exact:
            raise ValueError(f"Could not generate enough non-matching sequences after {max_attempts} attempts.")

        non_matches = [(seq, False) for seq in list(non_matches_set)]

        # Create separate test sets for exact and approximate matches
        exact_test_data = exact_matches + non_matches
        approx_test_data = approximate_matches + non_matches

        return exact_test_data, approx_test_data

    def mutate_sequence(self, sequence, num_mutations):
        """
        Mutate a given DNA sequence by changing a specified number of nucleotides.

        Parameters:
        - sequence (str): The original DNA sequence.
        - num_mutations (int): Number of nucleotides to mutate.

        Returns:
        - mutated_seq (str): The mutated DNA sequence.
        """
        sequence = list(sequence)
        length = len(sequence)
        mutation_indices = random.sample(range(length), min(num_mutations, length))

        for idx in mutation_indices:
            original_nuc = sequence[idx]
            possible_mutations = ['A', 'C', 'G', 'T']
            possible_mutations.remove(original_nuc)
            sequence[idx] = random.choice(possible_mutations)

        return ''.join(sequence)

    def encode_sequences(self, sequences, n):
        """
        Encode a list of DNA sequences into hypervectors.

        Parameters:
        - sequences (list of str): DNA sequences to encode.
        - n (int): Subword length for encoding.

        Returns:
        - encoded_hypervectors (list of numpy arrays): Hypervectors for each sequence.
        """
        subword_groups = []
        for seq in sequences:
            subwords = []
            for i in range(0, len(seq), n):
                subword = seq[i:i + n]
                if len(subword) == n:
                    subwords.append(subword)
            if subwords:
                subword_groups.append(subwords)
            else:
                subword_groups.append([])

        # Encode the subword groups into hypervectors
        encoded_hypervectors = self.create_encoded_library(subword_groups)

        return encoded_hypervectors

    def compute_hamming_distance(self, hv1, hv2):
        """
        Compute the Hamming distance between two hypervectors.

        Parameters:
        - hv1, hv2 (numpy arrays): Binary hypervectors.

        Returns:
        - distance (int): Number of differing bits.
        """
        return np.sum(hv1 != hv2)

    def search_hamming(self, test_hv, genome_hvs, threshold):
        """
        Search for a test hypervector in the genome hypervectors using Hamming distance.

        Parameters:
        - test_hv (numpy array): Hypervector of the test sequence.
        - genome_hvs (list of numpy arrays): Hypervectors of the genome sequences.
        - threshold (int): Maximum Hamming distance to consider a match.

        Returns:
        - is_match (bool): True if a match is found within the threshold, else False.
        """
        for genome_hv in genome_hvs:
            distance = self.compute_hamming_distance(test_hv, genome_hv)
            if distance <= threshold:
                return True
        return False

    def search_cosine(self, test_hv, genome_hvs, threshold):
        """
        Search for a test hypervector in the genome hypervectors using Cosine similarity.

        Parameters:
        - test_hv (numpy array): Hypervector of the test sequence.
        - genome_hvs (list of numpy arrays): Hypervectors of the genome sequences.
        - threshold (float): Minimum cosine similarity to consider a match.

        Returns:
        - is_match (bool): True if a match is found within the threshold, else False.
        """
        # Reshape for sklearn's cosine_similarity
        test_hv_reshaped = test_hv.reshape(1, -1)
        genome_matrix = np.array(genome_hvs)
        similarities = cosine_similarity(test_hv_reshaped, genome_matrix)
        max_similarity = np.max(similarities)
        return max_similarity >= threshold

    def evaluate_accuracy(self, encoded_test_hvs, labels, genome_hvs, threshold, method='hamming'):
        """
        Evaluate the accuracy of the search against the test labels.

        Parameters:
        - encoded_test_hvs (list of numpy arrays): Hypervectors for test sequences.
        - labels (list of bool): True labels for the test sequences.
        - genome_hvs (list of numpy arrays): Hypervectors for the genome sequences.
        - threshold: Threshold value for the similarity measure.
        - method (str): 'hamming' or 'cosine' to specify the similarity measure.

        Returns:
        - accuracy (float): Classification accuracy.
        - confusion_matrix (dict): Counts of TP, TN, FP, FN.
        """
        predictions = []
        confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for test_hv, true_label in zip(encoded_test_hvs, labels):
            if method == 'hamming':
                prediction = self.search_hamming(test_hv, genome_hvs, threshold)
            elif method == 'cosine':
                prediction = self.search_cosine(test_hv, genome_hvs, threshold)
            else:
                raise ValueError("Unsupported method. Choose 'hamming' or 'cosine'.")

            predictions.append(prediction)

            if prediction and true_label:
                confusion_matrix['TP'] += 1
            elif not prediction and not true_label:
                confusion_matrix['TN'] += 1
            elif prediction and not true_label:
                confusion_matrix['FP'] += 1
            elif not prediction and true_label:
                confusion_matrix['FN'] += 1

        correct = confusion_matrix['TP'] + confusion_matrix['TN']
        total = len(labels)
        accuracy = correct / total if total > 0 else 0

        return accuracy, confusion_matrix

def read_seq(inputfile):
    with open(inputfile, "r") as f:
        seq = f.read()
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    return seq


dna_seq = read_seq("genomic.txt")
# Example Usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # # Example genomic data
    # genomic_data = (
    #         "ATTAACATACCAACCATAATAACATTAACATACCAACCATAATAACATTAACATACCAACCATAATAACATTAACATACCAACCATAATAAC" * 5
    # )  # Repeat to increase length and unique substrings
    #
    dna_sequence = dna_seq

    hypervector_dim = 2000


    dna_query_len = 50
    subword_length = 10  # Must divide dna_query_len

    # Initialize BioHD instance
    test_instance = BioHD(
        hypervector_dimension=hypervector_dim,
        dna_sequence=dna_sequence,
        dna_query_length=dna_query_len
    )

    # Encode the entire genome
    chunks_divide_n = test_instance.ProteinSequenceEncoding(n=subword_length)
    genome_hvs = test_instance.create_encoded_library(chunks_divide_n)

    # Generate test data for exact matches and approximate matches

    num_exact = 40
    num_approx = 40
    num_mutations = 3
    # Number of mutations for approximate matches


    exact_test_data, approx_test_data = test_instance.create_test_data(
        num_exact=num_exact, num_approx=num_approx, num_mutations=num_mutations
    )

    # Separate exact match data and non-match data
    print(f"Generated {len(exact_test_data)} exact test sequences (exact matches + non-matches).")
    print(f"Generated {len(approx_test_data)} approximate test sequences (approximate matches + non-matches).\n")

    # Encode test data for exact matches
    exact_sequences = [seq for seq, label in exact_test_data]
    encoded_exact_hv = test_instance.encode_sequences(exact_sequences, n=subword_length)

    # Encode test data for approximate matches
    approx_sequences = [seq for seq, label in approx_test_data]
    encoded_approx_hv = test_instance.encode_sequences(approx_sequences, n=subword_length)

    # Define thresholds
    hamming_threshold = 200
    cosine_threshold = 0.77
    # Example threshold; adjust as needed

    # # Evaluate accuracy for exact matches (Hamming Distance)
    # accuracy_hamming_exact, confusion_hamming_exact = test_instance.evaluate_accuracy(
    #     encoded_test_hvs=encoded_exact_hv,
    #     labels=[label for _, label in exact_test_data],
    #     genome_hvs=genome_hvs,
    #     threshold=hamming_threshold,
    #     method='hamming'
    # )

    # # Evaluate accuracy for exact matches (Cosine Similarity)
    # accuracy_cosine_exact, confusion_cosine_exact = test_instance.evaluate_accuracy(
    #     encoded_test_hvs=encoded_exact_hv,
    #     labels=[label for _, label in exact_test_data],
    #     genome_hvs=genome_hvs,
    #     threshold=cosine_threshold,
    #     method='cosine'
    # )
    #
    # # Evaluate accuracy for approximate matches (Hamming Distance)
    # accuracy_hamming_approx, confusion_hamming_approx = test_instance.evaluate_accuracy(
    #     encoded_test_hvs=encoded_approx_hv,
    #     labels=[label for _, label in approx_test_data],
    #     genome_hvs=genome_hvs,
    #     threshold=hamming_threshold,
    #     method='hamming'
    # )

    # Evaluate accuracy for approximate matches (Cosine Similarity)
    accuracy_cosine_approx, confusion_cosine_approx = test_instance.evaluate_accuracy(
        encoded_test_hvs=encoded_approx_hv,
        labels=[label for _, label in approx_test_data],
        genome_hvs=genome_hvs,
        threshold=cosine_threshold,
        method='cosine'
    )

    # # Print results for Exact Matches (Hamming and Cosine)
    # print(f"Exact Matches - Hamming Distance - Accuracy: {accuracy_hamming_exact * 100:.2f}%")
    # print("Exact Matches Confusion Matrix (Hamming):")
    # for key, value in confusion_hamming_exact.items():
    #     print(f"{key}: {value}")
    #
    # print(f"Exact Matches - Cosine Similarity - Accuracy: {accuracy_cosine_exact * 100:.2f}%")
    # print("Exact Matches Confusion Matrix (Cosine):")
    # for key, value in confusion_cosine_exact.items():
    #     print(f"{key}: {value}")
    #
    # # Print results for Approximate Matches (Hamming and Cosine)
    # print(f"\nApproximate Matches - Hamming Distance - Accuracy: {accuracy_hamming_approx * 100:.2f}%")
    # print("Approximate Matches Confusion Matrix (Hamming):")
    # for key, value in confusion_hamming_approx.items():
    #     print(f"{key}: {value}")

    print(f"Approximate Matches - Cosine Similarity - Accuracy: {accuracy_cosine_approx * 100:.2f}%")
    print("Approximate Matches Confusion Matrix (Cosine):")
    for key, value in confusion_cosine_approx.items():
        print(f"{key}: {value}")
