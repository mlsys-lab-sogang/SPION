
listops = {
              "dataset":{
                  "train":96000,
                  "dev":2000,
                  "test":2000,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":64, 
                  "transformer_dim":64, 
                  "transformer_hidden_dim":128, 
                  "head_dim":32, 
                  "num_head":2, 
                  "num_layers":4,
                  "vocab_size":32,
                  "max_seq_len":2000,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes":10,
                  "block_size":64,
                  "batch_size":256, 
                  "density" : 0.04,
                  "threshold" : 0.98,
              },
              "training":{
                  "batch_size":256, 
                  "learning_rate":0.0007,
                  "warmup":1000,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":500, 
                  "num_train_steps":30000,
                  "num_init_steps":1000,
                  "num_eval_steps":62,
                  "num_dense_train_steps":2500,
                  "patience":10, 
              }
          }

pathfinder = {
           "model":{
               "learn_pos_emb":True,
               "tied_weights":False,
               "embedding_dim":64, 
               "transformer_dim":64, 
               "transformer_hidden_dim":128, 
               "head_dim":32,
               "num_head":2, 
               "num_layers":2,
               "vocab_size":512,
               "max_seq_len":1024,
               "dropout_prob":0.1,
               "attention_dropout":0.1,
               "pooling_mode":"MEAN",
               "num_classes": 2,
               "block_size":32,
               "batch_size":256, 
               "density" : 0.05,
               "threshold" : 0.96,
           },
           "training":{
               "batch_size":256, 
               "learning_rate":0.0003,
               "warmup":312, 
               "lr_decay":"linear",
               "weight_decay":0,
               "eval_frequency":500, 
               "num_train_steps":50000, 
               "num_init_steps":3500,
               "num_eval_steps":156, 
               "num_dense_train_steps":7000,
               "patience":10, 
           }
       }

retrieval={
              "dataset":{
                  "train":147086,
                  "dev":18090,
                  "test":17437,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":64, 
                  "transformer_dim":64, 
                  "transformer_hidden_dim":128, 
                  "head_dim":32, 
                  "num_head":2, 
                  "num_layers":2,
                  "vocab_size":512,
                  "max_seq_len":4000,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "block_size":64,
                  "num_classes": 2,
                  "batch_size":32,
                  "density" : 0.021,
                  "threshold" : 0.99,
              },
              "training":{
                  "batch_size":32, 
                  "learning_rate":0.0003,
                  "warmup":800,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":200,
                  "num_train_steps":40000, 
                  "num_init_steps":3000,
                  "num_eval_steps":300, 
                  "num_dense_train_steps":2500,
                  "patience":10, 
              }
}

image={
        "dataset":{
            "train":45000,
            "dev":5000,
            "test":10000,
        },
        "model":{
            "learn_pos_emb":True,
            "tied_weights":False,
            "embedding_dim":64,
            "transformer_dim":64,
            "transformer_hidden_dim":128,
            "head_dim":32,
            "num_head":2,
            "num_layers":2,
            "vocab_size":256, 
            "max_seq_len":1024,
            "dropout_prob":0.1, 
            "attention_dropout":0.1,
            "pooling_mode":"MEAN",
            "num_classes": 10,
            "block_size":32,
            "batch_size": 128, 
            "density" : 0.04,
            "threshold" : 0.96,
        },
        "training":{
            "batch_size":128, 
            "learning_rate":0.0001, 
            "warmup":175,
            "lr_decay":"linear",
            "weight_decay":0,
            "eval_frequency":500,  
            "num_train_steps":10000, 
            "num_init_steps":0,
            "num_eval_steps":20,
            "num_dense_train_steps":3000,
        }
}

text={
         "dataset":{
             "train":25000,
             "dev":25000,
             "test":25000,
         },
         "model":{
             "learn_pos_emb":True,
             "tied_weights":False,
             "embedding_dim":64, 
             "transformer_dim":64, 
             "transformer_hidden_dim":128, 
             "head_dim":32, 
             "num_head":2, 
             "num_layers":2,
             "vocab_size":512,
             "max_seq_len":4000, 
             "dropout_prob":0.1,
             "attention_dropout":0.1,
             "pooling_mode":"MEAN",
             "num_classes": 2,
             "block_size":64,
             "batch_size":32,
             "density" : 0.03,
             "threshold" : 0.99,
         },
         "training":{
             "batch_size":32,
             "learning_rate":0.0001,
             "warmup":80, 
             "lr_decay":"linear",
             "weight_decay":0,
             "eval_frequency":200, 
             "num_train_steps":30000,
             "num_init_steps":3000,
             "num_eval_steps":200, 
             "num_dense_train_steps":1000,
             "patience":10, 
         }
     }

Config = {
    "lra-listops":listops,
    "lra-pathfinder":pathfinder,
    "lra-retrieval":retrieval,
    "lra-image":image,
    "lra-text":text
}

Config["lra-pathfinder32-curv_contour_length_14"] = Config["lra-pathfinder"]