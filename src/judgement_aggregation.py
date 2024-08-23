## implement part 1 here
import numpy as np
import pandas as pd


class JudgementAggregator:

    def __init__(self, judgements):

        self.judgements = judgements
        self.mapping_relevance_binary = {
            "0_NOT_RELEVANT": 0,
            "1_TOPIC_RELEVANT_DOES_NOT_ANSWER": 0,
            "2_GOOD_ANSWER": 1,
            "3_PERFECT_ANSWER": 1,
        }
        self.mapping_relevance_multi = {
            "0_NOT_RELEVANT": 0,
            "1_TOPIC_RELEVANT_DOES_NOT_ANSWER": 1,
            "2_GOOD_ANSWER": 2,
            "3_PERFECT_ANSWER": 3,
        }

        # self.pairs = self.__calculate_pairs_flags()

    @staticmethod
    def __calc_flag_is_contradictory(judgements, mapping_relevance_binary):
        """
        This function calculates the flag of each row in the judgements dataframe
        if the row is contradictory or not. A row is contradictory if the number of
        unique values in the row is greater than 1.
        """
        judgements_binary = list(
            judgements["relevance_class"].map(mapping_relevance_binary)
        )
        return int(0 in judgements_binary and 1 in judgements_binary)

    @staticmethod
    def __calc_binary_relevance(judgements, mapping_relevance_binary):
        return (
            judgements["relevance_class"]
            .map(mapping_relevance_binary)
            .value_counts()
            .idxmax()
        )

    @staticmethod
    def __calc_has_majority(judgements):
        """
        Returns 1 if majority was achieved, 0 otherwise.
        """
        if len(judgements) == 1:
            return 1
        elif len(judgements) == 2:
            # check if both judgements are the same
            return int(judgements["relevance_class"].nunique() == 1)
        else:
            # check if the majority is at least 2
            return int(judgements["relevance_class"].value_counts().max() >= 2)

    @staticmethod
    def __calc_is_unanimoous(judgements):
        """
        Returns 1 if all judgements are the same, 0 otherwise.
        """
        if len(judgements) == 1:
            return 1
        elif len(judgements) == 2:
            # check if both judgements are the same
            return int(judgements.nunique() == 1)
        else:
            # check if all judgements are the same
            return int(judgements.nunique() == 1)

    @staticmethod
    def __calc_majority_voting_agg(judgements):
        """
        Returns the majority voting aggregation of the judgements.
        """
        return judgements["relevance_class"].value_counts().idxmax()

    @staticmethod
    def has_majority_coannot(judgements):
        has_majority = 0
        if len(judgements["relevance_class_coannot"]) == 1:
            has_majority = 1
        elif len(judgements["relevance_class_coannot"]) == 2:
            # check if both judgements are the same
            has_majority = int(judgements["relevance_class_coannot"].nunique() == 1)
        else:
            # check if the majority is at least 2
            has_majority = int(
                judgements["relevance_class_coannot"].value_counts().max() >= 2
            )
        return has_majority

    @staticmethod
    def has_alignment_with_majority(judgements):
        annot_relevance_class = (
            judgements["relevance_class_annot"].value_counts().idxmax()
        )
        coannot_relenvance_class = (
            judgements["relevance_class_coannot"].value_counts().idxmax()
        )
        aligns_with_majority = annot_relevance_class == coannot_relenvance_class
        return int(aligns_with_majority)

    def get_agreement_score(self):
        # get unique users
        annotators = self.judgements["user_id"].unique()
        # get all documents labeled by user
        total_pairs_with_majority = 0
        total_pairs_with_alignment = 0
        total_pairs_with_majority_and_alignment = 0
        total_pairs = 0
        df_user_agreement = pd.DataFrame(
            columns=[
                "user_id",
                "agreement_score",
                "has_majority",
                "has_alignment",
                "has_majority_and_alignment",
                "total_pairs",
            ]
        )
        for annotator in annotators:
            # get all judgements labeled by user
            df_annot_judgements = self.judgements[
                self.judgements["user_id"] == annotator
            ]
            df_coannot_judgements = self.judgements[
                ["query_id", "doc_id", "user_id", "relevance_class"]
            ].merge(
                df_annot_judgements[
                    ["query_id", "doc_id", "user_id", "relevance_class"]
                ],
                on=["query_id", "doc_id"],
                suffixes=("_coannot", "_annot"),
            )
            df_coannot = (
                df_coannot_judgements.groupby(by=["query_id", "doc_id"])
                .count()
                .reset_index()
            )

            df_coannot["has_majority"] = (
                df_coannot_judgements.groupby(by=["query_id", "doc_id"])
                .apply(self.has_majority_coannot)
                .reset_index()[0]
            )
            df_coannot["alignment_with_majority"] = (
                df_coannot_judgements.groupby(by=["query_id", "doc_id"])
                .apply(self.has_alignment_with_majority)
                .reset_index()[0]
            )
            df_coannot["has_majority_and_alignment"] = (
                df_coannot["has_majority"] & df_coannot["alignment_with_majority"]
            )

            total_pairs_with_majority += df_coannot["has_majority"].sum()
            total_pairs_with_alignment += df_coannot["alignment_with_majority"].sum()
            total_pairs_with_majority_and_alignment += df_coannot[
                "has_majority_and_alignment"
            ].sum()
            total_pairs += df_annot_judgements.shape[0]

            agreement_score_user = (
                df_coannot["has_majority_and_alignment"].sum()
                / df_coannot["has_majority"].sum()
            )
            user_row = {
                "user_id": annotator,
                "agreement_score": agreement_score_user,
                "has_majority": df_coannot["has_majority"].sum(),
                "has_alignment": df_coannot["alignment_with_majority"].sum(),
                "has_majority_and_alignment": df_coannot[
                    "has_majority_and_alignment"
                ].sum(),
                "total_pairs": df_annot_judgements.shape[0],
            }
            df_user_agreement = df_user_agreement.append(user_row, ignore_index=True)

            print(df_coannot.shape)
            print(user_row)

        return (
            df_user_agreement,
            total_pairs_with_majority,
            total_pairs_with_alignment,
            total_pairs_with_majority_and_alignment,
            total_pairs,
        )

    @staticmethod
    def relevance_present(judgements):
        return (
            judgements[
                judgements["relevance_character_ranges"] != "<no ranges selected>"
            ].shape[0]
            / judgements.shape[0]
        )

    @staticmethod
    def percentage_relevance_classes(judgements):
        return judgements["relevance_class"].value_counts()

    def __calculate_pairs_flags(self):
        flag_is_contradictory = (
            self.judgements.groupby(by=["query_id", "doc_id"])
            .apply(
                lambda x: self.__calc_flag_is_contradictory(
                    x, self.mapping_relevance_binary
                )
            )
            .reset_index()
        )
        flag_binary_relevance = (
            self.judgements.groupby(by=["query_id", "doc_id"])
            .apply(
                lambda x: self.__calc_binary_relevance(x, self.mapping_relevance_binary)
            )
            .reset_index()
        )
        flag_has_majority = (
            self.judgements.groupby(by=["query_id", "doc_id"])
            .apply(lambda x: self.__calc_has_majority(x))
            .reset_index()
        )
        flag_is_unanimoous = (
            self.judgements.groupby(by=["query_id", "doc_id"])
            .apply(lambda x: self.__calc_is_unanimoous(x["relevance_class"]))
            .reset_index()
        )

        df_pairs = (
            self.judgements.groupby(by=["query_id", "doc_id"])
            .count()["id"]
            .reset_index()
        )
        df_pairs = df_pairs.merge(flag_is_contradictory, on=["query_id", "doc_id"])
        df_pairs = df_pairs.merge(flag_binary_relevance, on=["query_id", "doc_id"])
        df_pairs = df_pairs.merge(flag_has_majority, on=["query_id", "doc_id"])
        df_pairs = df_pairs.merge(flag_is_unanimoous, on=["query_id", "doc_id"])

        df_pairs.columns = [
            "query_id",
            "doc_id",
            "judgement_count",
            "contradictory",
            "relevant",
            "has_majority",
            "is_unanimoous",
        ]
        df_pairs["contradictory_text"] = df_pairs["contradictory"].map(
            {0: "non-contradictory", 1: "contradictory"}
        )
        df_pairs["relevant_text"] = df_pairs["relevant"].map(
            {0: "non-relevant", 1: "relevant"}
        )
        df_pairs["has_majority_text"] = df_pairs["has_majority"].map(
            {0: "no majority", 1: "has majority"}
        )
        df_pairs["unanimous_text"] = df_pairs["is_unanimoous"].map(
            {0: "not unanimous", 1: "unanimous"}
        )
        df_pairs["aggregated_relevance_majority"] = df_pairs.apply(
            self.__calc_majority_voting_agg
        ).reset_index()
        return df_pairs

    def __calculate_agreement_score_users(self):
        (
            df_user_agreement,
            total_pairs_with_majority,
            total_pairs_with_alignment,
            total_pairs_with_majority_and_alignment,
            total_pairs,
        ) = self.get_agreement_score()
        df_users = (
            self.judgements.groupby(by=["user_id"])
            .agg({"id": "count", "duration_to_judge": "mean"})
            .reset_index()
        )
        df_users = df_users.merge(df_user_agreement, on="user_id")
        df_users["agreement_score"] = (
            df_users["agreement_score"] - df_users["agreement_score"].min()
        ) / (df_users["agreement_score"].max() - df_users["agreement_score"].min())
        # df_users.drop(columns=[0], inplace=True)
        return df_users

    @staticmethod
    def get_weighted_labels(judgements, mapping_relevance_multi):
        judgements_binary = list(
            judgements["relevance_class"].map(mapping_relevance_multi)
        )
        has_majority = 0
        is_contradictory = int(0 in judgements_binary and 1 in judgements_binary)

        # Is ranges are provided
        if len(judgements["relevance_class"]) == 1:
            has_majority = 1
        elif len(judgements["relevance_class"]) == 2:
            # check if both judgements are the same
            has_majority = int(judgements["relevance_class"].nunique() == 1)
        else:
            # check if the majority is at least 2
            has_majority = int(judgements["relevance_class"].value_counts().max() >= 2)
        ranges_provided = {
            "0_NOT_RELEVANT": 0,
            "1_TOPIC_RELEVANT_DOES_NOT_ANSWER": 0,
            "2_GOOD_ANSWER": 0,
            "3_PERFECT_ANSWER": 0,
        }
        for index, row in judgements.iterrows():
            if row["relevance_character_ranges"] != "<no ranges selected>":
                ranges_provided[row["relevance_class"]] += 1
        ranges_provided_flag = sum(ranges_provided.values()) > 0
        # j1, j2, j3 -> (j1 - 0.45, )

        # get first digit of the string in the column relevance_class

        if (
            has_majority
        ):  # majority achieved in judgements: (j1, j1, j1) or (j1, j1, j2)
            if (
                is_contradictory
            ):  # if there is a contradiction in judgements [j0, j1] vs [j2, j3]: (j0, j1, j2) or (j1, j3, j3)
                if (
                    ranges_provided_flag
                ):  # if there are ranges provided, return the relevance class with the highest count of judgements with ranges provided
                    return max(ranges_provided, key=ranges_provided.get)
                else:  # if there are no ranges provided, return the simple majority - return the simple majority
                    return judgements["relevance_class"].value_counts().idxmax()
            else:  # if there is no contradiction in judgements: (j0, j1, j1) or (j2, j2, j2) or (j2, j3, j3) - return the simple majority
                return judgements["relevance_class"].value_counts().idxmax()
        else:  # majority not achieved in judgements: (j0, j1, j2) or (j0, j1, j3) - return the agreement weighted majority
            # return judgements.groupby(by = ['relevance_class']).apply(lambda x: x['agreement_score'].sum()/ len(x['agreement_score'])).idxmax()
            return (
                judgements.groupby(by=["relevance_class"])
                .apply(lambda x: x["agreement_score"].sum())
                .idxmax()
            )

    def get_weighted_aggregated_judgements(self):
        self.agreement_score_users = self.__calculate_agreement_score_users()
        df_raw_judgements = self.judgements.merge(
            self.agreement_score_users[["user_id", "agreement_score"]], on="user_id"
        )
        df_raw_judgements.rename(
            columns={"agreement_score_x": "agreement_score"}, inplace=True
        )
        labels_weighted_majority = (
            df_raw_judgements.groupby(by=["query_id", "doc_id"])
            .apply(lambda x: self.get_weighted_labels(x, self.mapping_relevance_multi))
            .reset_index()
        )
        return labels_weighted_majority
