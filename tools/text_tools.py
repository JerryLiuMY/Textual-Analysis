def join_tt(df_rich):
    """ join title & text to article
    :param df_rich: enriched dataframe
    :return: combined title and text
    """

    if df_rich["title"] == "nan":
        art = df_rich["text"]
    else:
        art = " ".join([df_rich["title"], df_rich["text"]])

    return art
