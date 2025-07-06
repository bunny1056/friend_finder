import math
from collections import defaultdict
from typing import List, Dict, Set, Tuple


class FriendGrouper:
    def __init__(self, users: List[Dict[str, List[str]]]):
        self.users = users
        self.user_ids = list(range(len(users)))
        self.trait_sets = self._build_trait_sets()
        self.trait_weights = self._calculate_trait_weights()
        self.similarity_matrix = self._calculate_similarity()

    def _build_trait_sets(self) -> Dict[int, Set[str]]:
        """Convert user traits to sets for efficient comparison"""
        return {i: set(user['traits']) for i, user in enumerate(self.users)}

    def _calculate_trait_weights(self) -> Dict[str, float]:
        """
        Calculate TF-IDF inspired weights for traits.
        Rare traits get higher weights.
        """
        trait_counts = defaultdict(int)
        for traits in self.trait_sets.values():
            for trait in traits:
                trait_counts[trait] += 1
        
        total_users = len(self.users)
        return {
            trait: math.log(total_users / count) 
            for trait, count in trait_counts.items()
        }

    def _calculate_similarity(self) -> Dict[int, Dict[int, float]]:
        """
        Compute weighted similarity between all user pairs.
        Similarity = sum of weights of shared traits.
        """
        similarity = defaultdict(dict)
        for i in self.user_ids:
            for j in self.user_ids:
                if i != j:
                    shared_traits = self.trait_sets[i] & self.trait_sets[j]
                    similarity[i][j] = sum(self.trait_weights[t] for t in shared_traits)
        return similarity

    def _find_most_similar_pair(self, available: Set[int]) -> Tuple[int, int]:
        """Find the pair of available users with highest similarity"""
        max_sim = -1
        best_pair = (None, None)
        
        for i in available:
            for j in available:
                if i != j and self.similarity_matrix[i][j] > max_sim:
                    max_sim = self.similarity_matrix[i][j]
                    best_pair = (i, j)
        return best_pair

    def _expand_group(self, seed: int, available: Set[int], max_size: int) -> List[int]:
        """
        Grow a group by iteratively adding the most similar available user
        
        Args:
            seed: Starting user ID
            available: Set of ungrouped user IDs
            max_size: Maximum group size
            
        Returns:
            List of user IDs in the formed group
        """
        group = [seed]
        available.remove(seed)
        
        while len(group) < max_size and available:
            # Find user with highest average similarity to group members
            best_candidate = None
            best_score = -1
            
            for candidate in available:
                avg_similarity = sum(
                    self.similarity_matrix[member][candidate] 
                    for member in group
                ) / len(group)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_candidate = candidate
            
            if best_candidate:
                group.append(best_candidate)
                available.remove(best_candidate)
            else:
                break
                
        return group

    def generate_groups(self, max_group_size: int = 5) -> List[List[str]]:
        available = set(self.user_ids)
        groups = []
        
        while available:
            if len(available) >= 2:
                # Start group with most similar pair
                seed1, seed2 = self._find_most_similar_pair(available)
                group = [seed1, seed2]
                available.remove(seed1)
                available.remove(seed2)
            else:
                # Handle last remaining user
                group = [available.pop()]
                
            # Expand the group
            if len(group) < max_group_size and available:
                group = self._expand_group(group[0], available, max_group_size)
            
            groups.append(group)
        
        return [[self.users[i]['name'] for i in group] for group in groups]


def load_sample_users() -> List[Dict[str, List[str]]]:
    """Sample user data for demonstration"""
    return [
        {"name": "Alice", "traits": ["hiking", "reading", "sci-fi", "music", "introvert", "analytical"]},
        {"name": "Bob", "traits": ["sports", "music", "travel", "extrovert", "energetic"]},
        {"name": "Charlie", "traits": ["hiking", "music", "gaming", "ambivert", "creative"]},
        {"name": "Diana", "traits": ["reading", "art", "sci-fi", "introvert", "creative"]},
        {"name": "Eve", "traits": ["sports", "travel", "music", "extrovert", "logical"]},
        {"name": "Frank", "traits": ["gaming", "music", "tech", "introvert", "analytical"]},
        {"name": "Grace", "traits": ["art", "tech", "reading", "ambivert", "creative"]},
        {"name": "Heidi", "traits": ["sci-fi", "gaming", "music", "introvert", "curious"]},
        {"name": "Ivan", "traits": ["hiking", "travel", "sports", "extrovert", "adventurous"]},
        {"name": "Judy", "traits": ["reading", "tech", "gaming", "introvert", "logical"]}
    ]


def main():
    """Run the group generation with sample data"""
    users = load_sample_users()
    grouper = FriendGrouper(users)
    
    print("=== Friend Group Generator ===")
    max_size = int(input("Enter maximum group size (default 4): ") or 4)
    groups = grouper.generate_groups(max_group_size=max_size)
    
    print("\nOptimized Groups:")
    for i, group in enumerate(groups, 1):
        print(f"Group {i}: {', '.join(group)}")
    
    print("\nGrouping complete!")


if __name__ == "__main__":
    main()
