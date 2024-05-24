

def findPokerHand(hand):
    
    ranks = []
    suits = []
    
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
            
        if rank == 'A' : rank = 14
        elif rank == 'K' : rank = 13
        elif rank == 'Q' : rank = 12
        elif rank == 'J' : rank = 11
        
        ranks.append(int(rank))
        suits.append(suit)
        
    ranks.sort(reverse=True)
    suits.sort(reverse=True)
    print(hand)
    
    # suits
    isflush = (suits.count(suits[0]) == 5)
    
    # ranks
    rank_same = []
    pair_num = 0
    pairs = set()
    for rank in ranks:
        cnt =  ranks.count(rank)
        if cnt not in rank_same:
            rank_same.append(cnt)
        if cnt == 2:
            pairs.add(rank)
        pair_num = len(pairs)
    
    # straight
    isStraight = True
    for i in range(1, len(ranks)):
        if ranks[i] != ranks[i - 1] - 1:
            isStraight = False
    if not isStraight:
        isStraight = all(element in [14,5,4,3,2] for element in ranks)
    if not isStraight:
        isStraight = all(element in [11,12,13,14,2] for element in ranks)
    
    pokerHandRanks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 6: "Flush", 5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card"}
    # Hand Rank
    posible_ranks = []
    if isStraight and ranks[0] == 14 and ranks[1] == 13 and isflush:
        posible_ranks.append(10)
    if isStraight and isflush:
        posible_ranks.append(9)
    if 4 in rank_same:
        posible_ranks.append(8)
    if 3 in rank_same and pair_num == 1:
        posible_ranks.append(7)
    if isflush:
        posible_ranks.append(6)
    if isStraight:
        posible_ranks.append(5)
    if 3 in rank_same:
        posible_ranks.append(4)
    if pair_num == 2:
        posible_ranks.append(3)
    if pair_num == 1:
        posible_ranks.append(2)
    if len(posible_ranks) == 0:
        posible_ranks.append(1)
    
    for hand in posible_ranks:
        print(pokerHandRanks[hand])
        
    return pokerHandRanks[posible_ranks[0]]

if __name__ == "__main__":
    # findPokerHand(["KH", "AH", "QH", "JH", "10H"])  # Royal Flush
    # findPokerHand(["QC", "JC", "10C", "9C", "8C"])  # Straight Flush
    # findPokerHand(["5C", "5S", "5H", "5D", "QH"])  # Four of a Kind
    # findPokerHand(["2H", "2D", "2S", "10H", "10C"])  # Full House
    # findPokerHand(["2D", "KD", "7D", "6D", "5D"])  # Flush
    # findPokerHand(["JC", "10H", "9C", "8C", "7D"])  # Straight
    # findPokerHand(["10H", "10C", "10D", "2D", "5S"])  # Three of a Kind
    # findPokerHand(["KD", "KH", "5C", "5S", "6D"])  # Two Pair
    # findPokerHand(["2D", "2S", "9C", "KD", "10C"])  # Pair
    # findPokerHand(["KD", "5H", "2D", "10C", "JH"])  # High Card
    findPokerHand(['4H', '3H', '5H', 'AH', '2H'])
    pass