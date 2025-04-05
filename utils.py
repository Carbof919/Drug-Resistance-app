def preprocess_user_data(user_df, selected_genes):
    gene_expr = dict(zip(user_df.iloc[:, 0], user_df.iloc[:, 1]))  # {gene: expression}

    input_vector = []
    matched_count = 0

    for gene in selected_genes:
        if gene in gene_expr:
            input_vector.append(gene_expr[gene])
            matched_count += 1
        else:
            input_vector.append(0)  # or average value, or np.nan

    return input_vector, matched_count
