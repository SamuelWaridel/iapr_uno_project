import pandas as pd

def make_submission_csv(features, output_path="predicted_labels.csv"):
    """
    Creates a csv file for submission to kaggle.

    Args:
        features (dict): A dictionary containing image IDs as keys and feature dictionaries as values.
            Global structure of the features dictionary:
            {
                "image_id_1": {
                    "active_player": "player_1",
                    "cards": [
                        {"full_label": "r_5","player": "center", ...},
                        {"full_label": "b_skip", "player": "player_1", ...},
                        ...
                    ],
                    ...
        
        output_path (str, optional): The path where the CSV file will be saved. Defaults to "predicted_labels.csv".

    Returns:
        pd.DataFrame: The created DataFrame.
    """

    image_indexes = list(features.keys())
    center_card = []
    player_1_cards = []
    player_2_cards = []
    player_3_cards = []
    player_4_cards = []
    active_player = []

    for img_id, feat in features.items():
        active_player.append(feat["active_player"])
        
        current_center_card = None # variable to store the center card for the current image, because there should be only one center card per image and we have more than one sometimes
        current_player_1_card = None
        current_player_2_card = None
        current_player_3_card = None
        current_player_4_card = None
        
        player_1_card_shortlist = []
        player_2_card_shortlist = []
        player_3_card_shortlist = []
        player_4_card_shortlist = []
        
        for card in feat["cards"]:        
            if card["player"] == "center":
                if current_center_card is not None:
                    print(f"Warning: multiple center cards detected in image {img_id}")
                else: 
                    center_card.append(card["full_label"])
                    current_center_card = card["full_label"]
            elif card["player"] == "player_1":
                player_1_card_shortlist.append(card["full_label"])
                current_player_1_card = card["full_label"]
            elif card["player"] == "player_2":
                player_2_card_shortlist.append(card["full_label"])
                current_player_2_card = card["full_label"]
            elif card["player"] == "player_3":
                player_3_card_shortlist.append(card["full_label"])
                current_player_3_card = card["full_label"]
            elif card["player"] == "player_4":
                player_4_card_shortlist.append(card["full_label"])
                current_player_4_card = card["full_label"]

        if current_center_card is None:
            print(f"Warning: no center card detected in image {img_id}")
            center_card.append("EMPTY")
            
        if current_player_1_card is None:
            player_1_cards.append("EMPTY")
        else: 
            # concatenate all detected cards for player 1 with ";" as separator
            player_1_cards.append(";".join(player_1_card_shortlist))
            
        if current_player_2_card is None:
            player_2_cards.append("EMPTY")
        else: 
            player_2_cards.append(";".join(player_2_card_shortlist))    
            
        if current_player_3_card is None:
            player_3_cards.append("EMPTY")
        else:
            player_3_cards.append(";".join(player_3_card_shortlist))
            
        if current_player_4_card is None:
            player_4_cards.append("EMPTY")
        else:
            player_4_cards.append(";".join(player_4_card_shortlist))
            
        

    # create a dataframe with the extracted labels

    df = pd.DataFrame({
        "image_id": image_indexes,
        "center_card": center_card,
        "active_player": active_player,
        "player_1_cards": player_1_cards,
        "player_2_cards": player_2_cards,
        "player_3_cards": player_3_cards,
        "player_4_cards": player_4_cards
    })

    df.to_csv(output_path, index=False)
    
    return df