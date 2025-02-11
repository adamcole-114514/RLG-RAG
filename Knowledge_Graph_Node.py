class Knowledge_Graph_Node():
    def __init__(self, id, subquestion, knowledge, prev_nodes, next_nodes, if_sink):
        self.id = id
        self.subquestion = subquestion
        self.knowledge = knowledge
        self.prev_nodes = prev_nodes
        self.next_nodes = next_nodes
        self.if_sink = if_sink

    def __call__(self, *args, **kwargs):
        print("Node Information:\nid:{0}\nsub-question:{1}\nknowledge:{2}\nprev-nodes:{3}\nnext-nodes:{4}\nif-sink:{5}".format(self.id,self.subquestion,self.knowledge,self.prev_nodes,self.next_nodes,self.if_sink))

if __name__ == "__main__":
    root_node = Knowledge_Graph_Node(0, "what is the biggest ocean in the earth?", "The Pacific Ocean is the biggest ocean in the earth.",[],[],False)
    root_node()