import numpy as np
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM, Notears
import networkx as nx


class DAGCreator:
    def __init__(
        self,
        train_df,
        X_features,
        Y_feature,
        max_samples=1000,
        method="notears",
    ) -> None:

        self.train_df = train_df
        self.X_features = X_features
        self.Y_feature = Y_feature
        self.max_samples = max_samples
        self.method = method
        self.predefined_dags = {
            "jobs": self.load_predefined_jobs_icalingam,
            "toy_example": self.load_predefined_toy_example,
            "toy_example_2": self.load_predefined_toy_example_2,
            "toy_example_single": self.load_predefined_toy_example_single,
            "toy_example_2_single": self.load_predefined_toy_example_2_single,
            "toy_example_no_individual": self.load_predefined_toy_example_no_individual,
            "toy_example_2_no_individual": self.load_predefined_toy_example_2_no_individual,
            "toy_example_2_no_neutral": self.load_predefined_toy_example_2_no_neutral,
        }

    def run_gcastle(self, method):

        algo_dict = {
            "pc": PC,
            "ges": GES,
            "icalingam": ICALiNGAM,
            "golem": GOLEM,
            "notears": Notears,
        }
        algo = algo_dict[method]()
        algo.learn(
            self.train_df[self.X_features + self.Y_feature].sample(
                np.min([self.train_df.shape[0], self.max_samples])
            )
        )

        # Relabel the nodes
        learned_graph = nx.DiGraph(algo.causal_matrix)
        MAPPING = {
            k: v
            for k, v in zip(
                range(algo.causal_matrix.shape[0]), self.X_features + self.Y_feature
            )
        }
        learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)
        self.dag_edges = list(learned_graph.edges())

    def remove_target_from_dag_edges(self):

        new_edges = []
        for l in self.dag_edges:
            new_edge = []

            for i in l:
                if i != self.Y_feature[0]:
                    new_edge.append(i)

            # if len(new_edge) >= 2:
            new_edges.append(new_edge)

        self.dag_edges = new_edges

    def create_dag_edges(self, add_nodes_not_in_edges=False):

        if self.method in ["pc", "ges", "icalingam", "golem", "notears"]:
            self.run_gcastle(self.method)

        else:
            print("Method not supported")

        self.create_dag_constraints()
        self.remove_target_from_dag_edges()
        if add_nodes_not_in_edges:
            self.add_nodes_not_in_edges()

        return [list(set(d)) for d in self.dag_edges if len(d) > 0]

    def add_nodes_not_in_edges(self):
        nodes_in_edges = [item for sublist in self.dag_edges for item in sublist]
        nodes_not_in_edges = [x for x in self.X_features if x not in nodes_in_edges]
        self.dag_edges.append(nodes_not_in_edges)

    def create_dag_constraints(self):
        """Right now, it changes dag edges to subgroups of variables from the roots"""

        def get_parents(dag, node):
            # Added childs
            parents = []
            for origin, destination in dag:
                if destination == node:
                    parents.append(origin)
                # if origin == node:
                #     parents.append(destination)

            return parents

        # def get_children(dag, node):
        #     # Added childs
        #     children = []
        #     for origin, destination in dag:
        #         if origin == node:
        #             children.append(destination)

        #     return children

        def get_ancestors(dag, node):
            ancestors = []
            for origin, destination in dag:

                # Dont take into account ancestors that go through Y
                if destination == self.Y_feature[0]:
                    continue

                if destination == node:
                    ancestors.append(origin)
                    ancestors.extend(get_ancestors(dag, origin))
            return ancestors

        dag_edges = self.dag_edges
        # obtain target parent nodes
        target_parents = get_parents(dag_edges, self.Y_feature[0])
        # target_childs = get_children(dag_edges, self.Y_feature[0])
        dag_constraints = {
            parent: get_ancestors(dag_edges, parent) + [parent]
            for parent in target_parents
        }
        # Add parents group
        dag_constraints["parents"] = target_parents

        self.dag_constraints = dag_constraints
        # CAUTION, dag_edges is not dag edges anymore
        self.dag_edges = list(dag_constraints.values())

    def return_predefined_dag(self, dataset_name):
        return self.predefined_dags[dataset_name]()

    @staticmethod
    def load_predefined_jobs_icalingam():
        dag = [
            ["x9", "x11", "x15", "x12", "x8", "x7", "x2", "x6", "x0"],
            ["x2", "x7"],
            ["x8", "x2", "x9", "x0"],
            ["x8", "x2"],
            ["x8", "x2", "x9"],
            ["x9", "x11", "x15", "x1", "x12", "x8", "x7", "x6", "x0"],
            ["x9", "x11", "x15", "x12", "x8", "x7", "x6", "x0"],
            ["x1", "x7"],
            ["x16"],
        ]
        return dag

    @staticmethod
    def load_predefined_toy_example():
        dag = [
            ["col_0"],
            ["col_1"],
            ["col_0", "col_1", "T"],
            ["col_0", "col_1", "T"],
            # ["col_0", "T"],
            # ["col_0", "T"],
        ]
        return dag

    @staticmethod
    def load_predefined_toy_example_single():
        dag = [["col_0", "col_1", "T"]]
        return dag

    @staticmethod
    def load_predefined_toy_example_no_individual():
        dag = [
            ["col_0", "col_1", "T"],
            ["col_0", "col_1", "T"],
        ]
        return dag

    @staticmethod
    def load_predefined_toy_example_2():
        dag = [
            ["col_0"],
            ["col_1"],
            ["col_0", "col_1", "T"],
            # ["col_0", "col_1", "col_2", "col_3", "col_4", "T"],
            ["col_0", "col_1", "col_4", "T"],
            ["col_2", "col_3", "col_4"],
        ]
        return dag

    @staticmethod
    def load_predefined_toy_example_2_single():
        dag = [
            ["col_0", "col_1", "col_2", "col_3", "col_4", "T"],
        ]
        return dag

    @staticmethod
    def load_predefined_toy_example_2_no_individual():
        dag = [
            ["col_0", "col_1", "T"],
            ["col_0", "col_1", "col_4", "T"],
            ["col_2", "col_3", "col_4"],
        ]
        return dag

    @staticmethod
    def load_predefined_toy_example_2_no_neutral():
        dag = [
            # ["col_0"],
            # ["col_1"],
            ["col_0", "col_1", "T"],
            ["col_0", "col_1", "col_2", "col_3", "col_4", "T"],
            # ["col_0", "col_1", "col_4", "T"],
            # ["col_2", "col_3", "col_4"],
        ]
        return dag


    @staticmethod
    def load_predefined_causalml_mode_1_dag():
        dag = [
            ["col_0", "col_1", "T"],
            # ["col_0", "col_1", "col_2", "col_3", "col_4", "T"],
            ["col_2"],
            ["col_3"],
            ["col_4"],
            ["col_0", "col_1"],
            ["col_0"],
            ["col_1"],
        ]
        return dag

    @staticmethod
    def load_predefined_causalml_mode_2_dag():
        dag = [
            # ["col_0", "col_1", "col_2", "col_3", "col_4", "T"],
            ["col_2"],
            ["col_0", "col_1", "col_2"],
            ["col_3", "col_4"],
            ["col_3"],
            ["col_4"],
            ["col_0", "col_1"],
            ["col_0"],
            ["col_1"],
        ]
        return dag

    @staticmethod
    def load_predefined_causalml_mode_3_dag():
        dag = [
            ["col_0", "col_1", "col_2"],
            ["col_0"],
            ["col_1"],
            ["col_2"],
            ["col_1", "col_2", "T"],
        ]
        return dag

    @staticmethod
    def load_predefined_causalml_mode_4_dag():
        dag = [
            ["col_0", "col_1", "col_2"],
            ["col_3", "col_4"],
            ["col_0", "col_1", "col_2", "col_3", "col_4", "T"],
            ["col_0"],
            ["col_1"],
            ["col_2"],
            ["col_3"],
            ["col_4"],
        ]
        return dag
