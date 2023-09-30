from karateclub import Diff2Vec
from karateclub import Graph2Vec
import matplotlib.pyplot as plt
import networkx as nx
# We can change the model of autoencoder here.
from pyod.models.ecod import ECOD
import joblib
from sklearn.metrics import precision_recall_curve



def embedding_node(graph, save_model=False, filename="node_embedding_model.joblib"):
    embedding_model = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
    embedding_model.fit(graph)
    X = embedding_model.get_embedding()
    if save_model:
        joblib.dump(embedding_model, filename)
    return X

def embedding_gragh(graphs, save_model=False, filename="graphs_embedding_model.joblib"):
    # 使用 Graph2Vec 进行图形表示学习
    graph2vec = Graph2Vec()
    graph2vec.fit(graphs)

    # 获取图形的向量表示
    graph_vectors = graph2vec.get_embedding()
    if save_model:
        joblib.dump(graph2vec, filename)
    return graph_vectors

def ae_train(X, save_model=False, filename="autoencode.joblib"):
    clf = ECOD()
    clf.fit(X)
    if save_model:
        joblib.dump(clf, filename)
    return clf
        
def get_outlier_scores(X, model, use_save_model=False, filename=""):
    if use_save_model:
        try:
            model = joblib.load(filename)
        except:
            print("cannot fing the model.")
            return 0
    return model.decision_function(X)
    
def draw_score(score, label):
    # plot outlier scores for training data
    plt.plot(score, label=label)
    # add legend and title
    plt.legend()
    plt.title('Outlier Scores')
    # show the plot
    plt.show()
    
def graph2score_train(graphs):
    X = embedding_gragh(graphs)
    ae = ae_train(X)
    train_scores = ae.decision_scores_
    draw_score(train_scores, "train")
    return ae

if __name__ == "__main__":
    # 构建带有属性的图形数据集
    import pandas as pd
    import networkx as nx

    # 构建带有属性的图形数据集
    graphs = []

    # 循环遍历每个 CSV 文件
    for i in range(20000):
        # 构造 CSV 文件路径
        csv_file = f"./data/csv/{i}.csv"

        # 从 CSV 文件读取数据
        try:
            # 从 CSV 文件读取数据
            df = pd.read_csv(csv_file)

            # 创建新的图对象
            G = nx.DiGraph()

            # 遍历每一行数据
            for _, row in df.iterrows():
                start_node_id = row['start_node_id']
                end_node_id = row['end_node_id']
                start_node_hash = row['start_node_hash']
                end_node_hash = row['end_node_hash']
                edge_time = row['edge_time']
                start_node_type = row['start_node_type']
                end_node_type = row['end_node_type']
                edge_type =row['edge_type']
                try:
                    start_node_description = row['start_node_description']
                except KeyError:
                    start_node_description = 'NaN'
                try:
                    end_node_description = row['end_node_description']
                except KeyError:
                    end_node_description = 'NaN'
                try:
                    edge_operation = row['edge_operation']
                except KeyError:
                    edge_operation = 'NaN'



                G.add_node(start_node_id, type=start_node_type,dedescription=start_node_description)
                G.add_node(end_node_id, type=end_node_type,dedescription=end_node_description)
                # 添加边及其属性
                G.add_edge(start_node_id, end_node_id, time=edge_time,type=edge_type,operation=edge_operation)

            # 将图添加到列表中
            graphs.append(G)

        except FileNotFoundError:
            # 文件不存在，跳过并继续下一个文件
            continue

    # 将图的节点重新编号
    graphs = [nx.convert_node_labels_to_integers(graph, first_label=0) for graph in graphs]

    # 对图进行进一步处理或训练
    # 训练模型并获取训练集的异常得分
    # 训练模型并获取训练集的异常得分
    ae_model = graph2score_train(graphs)
    plt.hist(ae_model.decision_scores_, bins=50, density=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution')
    plt.show()





