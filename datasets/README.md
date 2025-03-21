# 0. MOBINS

If you want to use baseline code, please use `npy format` files.
- `npy format` dataset to train the baselines: [all npy files link](https://drive.google.com/file/d/1_kgT79BKyXKfJvmt8aYlSNF8i8ykSOLt/view?usp=sharing). If you want to download each dataset, please use the below links. 
   -  `Transportation-Seoul`: [npy file link](https://drive.google.com/file/d/14QfBDA_R-PFXdoad5oR4S_lxSFdveRgN/view?usp=sharing) 
   -  `Transportation-Busan`: [npy file link](https://drive.google.com/file/d/1yWhGQ1REHUO2u8C56CYIhfCYQh2WnpJt/view?usp=sharing) 
   -  `Transportation-Daegu`: [npy file link](https://drive.google.com/file/d/1dj68ZwJC9kr8VlU6fIdHtq46htUHQ_94/view?usp=sharing)
   -  `Transportation-NYC`: [npy file link](https://drive.google.com/file/d/1-UkDLGyXSwPyFwgfMeDyw2-zLIQyMEYD/view?usp=sharing) 
   -  `Epidemic-Korea`: [npy file link](https://drive.google.com/file/d/1HYH_M-gYTRBk58XVQtjJTKOosPV-ovn9/view?usp=sharing)
   -  `Epidemic-NYC`: [npy file link](https://drive.google.com/file/d/1m7dHXhyQ6khLLcJyWZg06L0uGLThV303/view?usp=sharing)




# 1. MOBINS: Transportation datasets

## Download for Transportation Datasets
- `csv format` data: If you want to download each dataset, please use the below links. 
   -  `Transportation-Seoul`: [csv file link](https://drive.google.com/file/d/1MPkqpoeysY6tF-LUuya-bai5V0Fygx77/view?usp=sharing) 
   -  `Transportation-Busan`: [csv file link](https://drive.google.com/file/d/1PU9aqp7kc4UzuFOAqLPkraM5MbmtyodM/view?usp=sharing) 
   -  `Transportation-Deagu`: [csv file link](https://drive.google.com/file/d/1ggcUHlpT7QCYB1jwlouhZLgM2FzDnuKU/view?usp=sharing) 
   -  `Transportation-NYC`: [csv file link](https://drive.google.com/file/d/1R97wE9_v2WVAS1MAWKUbZWOhPQNQYCnV/view?usp=sharing)
- `npy format` data: If you want to download each dataset, please use the below links.
   -  `Transportation-Seoul`: [npy file link](https://drive.google.com/file/d/14QfBDA_R-PFXdoad5oR4S_lxSFdveRgN/view?usp=sharing) 
   -  `Transportation-Busan`: [npy file link](https://drive.google.com/file/d/1yWhGQ1REHUO2u8C56CYIhfCYQh2WnpJt/view?usp=sharing) 
   -  `Transportation-Daegu`: [npy file link](https://drive.google.com/file/d/1dj68ZwJC9kr8VlU6fIdHtq46htUHQ_94/view?usp=sharing)
   -  `Transportation-NYC`: [npy file link](https://drive.google.com/file/d/1-UkDLGyXSwPyFwgfMeDyw2-zLIQyMEYD/view?usp=sharing) 

## Simple Metadata for Transportation Datasets
- `csv format` datasets in every environment: each dataset has three components.
   - `SPATIAL_NETWORK.csv`: ( $n * n$ where $n$ = # of nodes ) 
      - Column name list: INDEX, $N_{0}$, $N_{1}$, $\dots$, $N_{n}$ 
      - INDEX list: $N_{0}$, $N_{1}$, $\dots$, $N_{n}$
   - `NODE_TIME_SERIES_FEATURES.csv`:  ($t$ * $p$ ) * ( $n$ * $d$ ) where $t$ = # of timestamps in a day, $p$ = total period, and $d$ = # of variables from time series.
      -  Column name list( _Transportation-[Seoul, Busan, Deagu]_ ): datetime, $N_{0}$ \_INFLOW, $N_{0}$ \_OUTFLOW, $\dots$, $N_{n}$ \_INFLOW, $N_{n}$ \_OUTFLOW
      -  Column name list( _Transportation-NYC_ ): datetime, $N_{0}$ \_RIDERSHIP, $\dots$,  $N_{n}$ \_RIDERSHIP
   - `OD_MOVEMENTS.csv`:  ($t$ * $p$ ) * ( $n$, $n$ ) 
      - Column name list: $N_{0}$ _ $N_{0}$, $N_{0}$ _ $N_{1}$, $N_{0}$ _ $N_{2}$, $\dots$ , $N_{n}$ _ $N_{n-1}$ , $N_{n}$ _ $N_{n}$
- `npy format` datasets for directly training the models on the Python environments: each dataset has three components.
   - `adj_matrix`(shape: ( $n$ , $n$ )) where $n$ = # of nodes
   - `node_npy`(each file is constructed in a daily manner with shape: ( $n$, $t$ , $d$ )) where $t$ = # of timestamps in a day and $d$ = # of variables from time series.
   - `od_npy` (each file is constructed in a daily manner with shape: ( $n$, $n$, $t$ )) 

## Metadata for Transportation Datasets
If you want to use CSV format files, please read the below metadata for each dataset.

Each file contains information about a single node or a node pair, which is abstracted for simplicity by describing only the i-th node. We omit the detailed description in metadata for __Transportation-[Busan, Daegu]__ because the CSV file structures are identical to metadata for __Transportation_Seoul__, differing only in the number of nodes, which is unique to each dataset. __Transportation_NYC__ follows a similar structure, with the exception of the variable for node time-series features (ridership).

### Metadata for Transportation-Seoul dataset
```code
{
  "@type": "sc:Dataset",
  "name": "Transportation-Seoul",
  "description": "This dataset contains transportation data for Seoul, including spatial network data, node time series features, and origin-destination movements.",
  "license": "https://creativecommons.org/licenses/by-nc/4.0",
  "url": "https://drive.google.com/file/d/1MPkqpoeysY6tF-LUuya-bai5V0Fygx77/view?usp=sharing",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "SPATIAL_NETWORK.csv",
      "name": "SPATIAL_NETWORK.csv",
      "contentUrl": "Transportation-Seoul/SPATIAL_NETWORK.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "NODE_TIME_SERIES_FEATURES.csv",
      "name": "NODE_TIME_SERIES_FEATURES.csv",
      "contentUrl": "Transportation-Seoul/NODE_TIME_SERIES_FEATURES.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "OD_MOVEMENTS.csv",
      "name": "OD_MOVEMENTS.csv",
      "contentUrl": "Transportation-Seoul/OD_MOVEMENTS.csv",
      "encodingFormat": "text/csv"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "name": "SPATIAL_NETWORK",
      "description": "This table represents the spatial network of nodes in Seoul.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "INDEX",
          "description": "The index column containing node identifiers. ",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "SPATIAL_NETWORK.csv" },
            "extract": {
              "column": "INDEX"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni",
          "description": "Columns representing spatial connections between all nodes and i-th node in the spatial network to describe an adjacency matrix ",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "SPATIAL_NETWORK.csv" },
            "extract": {
              "column": "Ni"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "name": "NODE_TIME_SERIES_FEATURES",
      "description": "This table contains time series features for each node over multiple timestamps.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "datetime",
          "description": "The datetime column representing timestamps (year, month, day, hour).",
          "dataType": "sc:DateTime",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "datetime"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni_INFLOW",
          "description": "Columns representing inflow time series for i-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "Ni_INFLOW"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni_OUTFLOW",
          "description": "Columns representing outflow time series for i-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "Ni_OUTFLOW"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "name": "OD_MOVEMENTS",
      "description": "This table represents origin-destination movements between nodes.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "Ni_Nj",
          "description": "Columns representing movements from i-th node to j-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "OD_MOVEMENTS.csv" },
            "extract": {
              "column": "Ni_Nj"
            }
          }
        }
      ]
    }
  ]
}
```

### Metadata for Transportation-Busan dataset
```code
{
  "@type": "sc:Dataset",
  "name": "Transportation-Busan",
  "description": "This dataset contains transportation data for Busan, including spatial network data, node time series features, and origin-destination movements.",
  "license": "https://creativecommons.org/licenses/by-nc/4.0",
  "url": "https://drive.google.com/file/d/1PU9aqp7kc4UzuFOAqLPkraM5MbmtyodM/view?usp=sharing",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "SPATIAL_NETWORK.csv",
      "name": "SPATIAL_NETWORK.csv",
      "contentUrl": "Transportation-Busan/SPATIAL_NETWORK.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "NODE_TIME_SERIES_FEATURES.csv",
      "name": "NODE_TIME_SERIES_FEATURES.csv",
      "contentUrl": "Transportation-Busan/NODE_TIME_SERIES_FEATURES.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "OD_MOVEMENTS.csv",
      "name": "OD_MOVEMENTS.csv",
      "contentUrl": "Transportation-Busan/OD_MOVEMENTS.csv",
      "encodingFormat": "text/csv"
    }
  ]
}
```

### Metadata for Transportation-Daegu dataset
```code
{
  "@type": "sc:Dataset",
  "name": "Transportation-Daegu",
  "description": "This dataset contains transportation data for Daegu, including spatial network data, node time series features, and origin-destination movements.",
  "license": "https://creativecommons.org/licenses/by-nc/4.0",
  "url": "https://drive.google.com/file/d/1ggcUHlpT7QCYB1jwlouhZLgM2FzDnuKU/view?usp=sharing",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "SPATIAL_NETWORK.csv",
      "name": "SPATIAL_NETWORK.csv",
      "contentUrl": "Transportation-Daegu/SPATIAL_NETWORK.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "NODE_TIME_SERIES_FEATURES.csv",
      "name": "NODE_TIME_SERIES_FEATURES.csv",
      "contentUrl": "Transportation-Daegu/NODE_TIME_SERIES_FEATURES.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "OD_MOVEMENTS.csv",
      "name": "OD_MOVEMENTS.csv",
      "contentUrl": "Transportation-Daegu/OD_MOVEMENTS.csv",
      "encodingFormat": "text/csv"
    }
  ]
}
```
### Metadata for Transportation-NYC dataset
```code
{
  "@type": "sc:Dataset",
  "name": "Transportation-NYC",
  "description": "This dataset contains transportation data for NYC, including spatial network data, node time series features, and origin-destination movements.",
  "license": "https://creativecommons.org/licenses/by-nc/4.0",
  "url": "https://drive.google.com/file/d/1R97wE9_v2WVAS1MAWKUbZWOhPQNQYCnV/view?usp=sharing",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "SPATIAL_NETWORK.csv",
      "name": "SPATIAL_NETWORK.csv",
      "contentUrl": "Transportation-NYC/SPATIAL_NETWORK.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "NODE_TIME_SERIES_FEATURES.csv",
      "name": "NODE_TIME_SERIES_FEATURES.csv",
      "contentUrl": "Transportation-NYC/NODE_TIME_SERIES_FEATURES.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "OD_MOVEMENTS.csv",
      "name": "OD_MOVEMENTS.csv",
      "contentUrl": "Transportation-NYC/OD_MOVEMENTS.csv",
      "encodingFormat": "text/csv"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "name": "SPATIAL_NETWORK",
      "description": "This table represents the spatial network of nodes in Seoul.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "INDEX",
          "description": "The index column containing node identifiers.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "SPATIAL_NETWORK.csv" },
            "extract": {
              "column": "INDEX"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni",
          "description": "Columns representing spatial connections between all nodes and i-th node in the spatial network to describe an adjacency matrix ",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "SPATIAL_NETWORK.csv" },
            "extract": {
              "column": "Ni"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "name": "NODE_TIME_SERIES_FEATURES",
      "description": "This table contains time series features for each node over multiple timestamps.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "datetime",
          "description": "The datetime column representing timestamps (year, month, day, hour).",
          "dataType": "sc:DateTime",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "datetime"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni_RIDERSHIP",
          "description": "Columns representing ridership time series for i-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "Ni_RIDERSHIP"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "name": "OD_MOVEMENTS",
      "description": "This table represents origin-destination movements between nodes.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "Ni_Nj",
          "description": "Columns representing movements from i-th node to j-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "OD_MOVEMENTS.csv" },
            "extract": {
              "column": "Ni_Nj"
            }
          }
        }
      ]
    }
  ]
}
```


# 2. MOBINS: Epidemic datasets

## Download for Epidemic Datasets
- `csv format` data: If you want to download each dataset, please use the below links. 
   -  `Epidemic-Korea`: [csv file link](https://drive.google.com/file/d/1G0P4-HdRoU6X2p18VmOaEMJVPIUEEhiE/view?usp=sharing)
   -  `Epidemic-NYC`: [csv file link](https://drive.google.com/file/d/1ZGiXpG6JpSRLQLllv6CefnM62NVRZLtk/view?usp=sharing)
- `npy format` data: If you want to download each dataset, please use the below links. 
   -  `Epidemic-Korea`: [npy file link](https://drive.google.com/file/d/1HYH_M-gYTRBk58XVQtjJTKOosPV-ovn9/view?usp=sharing)
   -  `Epidemic-NYC`: [npy file link](https://drive.google.com/file/d/1m7dHXhyQ6khLLcJyWZg06L0uGLThV303/view?usp=sharing)

## Simple Metadata for Epidemic Datasets
- `csv format` datasets in every environment: each dataset has three components.
   - `SPATIAL_NETWORK.csv`: ( $n * n$ where $n$ = # of nodes ) 
      - Column name list: INDEX, $N_{0}$, $N_{1}$, $\dots$, $N_{n}$ 
      - INDEX list: $N_{0}$, $N_{1}$, $\dots$, $N_{n}$
   - `NODE_TIME_SERIES_FEATURES.csv`:  ( $t$ * $p$ ) * ( $n$ * $d$ ) where $t$ = # of timestamps in a day, $p$ = total period, and $d$ = # of variables from time series
      -  Column name list: datetime, $N_{0}$ \_INFECTION, $\dots$,  $N_{n}$ \_INFECTION
   - `OD_MOVEMENTS.csv`:  ($t$ * $p$ ) * ( $n$, $n$ ) 
      - Column name list: $N_{0}$ _ $N_{0}$, $N_{0}$ _ $N_{1}$, $N_{0}$ _ $N_{2}$, $\dots$ , $N_{n}$ _ $N_{n-1}$ , $N_{n}$ _ $N_{n}$
- `npy format` datasets for directly training the models on the Python environments: each dataset has three components.
   - `adj_matrix`(shape: ( $n$ , $n$ )) where $n$ = # of nodes
   - `node_npy`(each file is constructed in a daily manner with shape: ( $n$, $t$ , $d$ )) where $t$ = # of timestamps in a day and $d$ = # of variables from time series
   - `od_npy` (each file is constructed in a daily manner with shape: ( $n$, $n$, $t$ )) 

## Metadata for Epidemic Datasets
If you want to use CSV format files, please read the below metadata for each dataset.

Each file contains information about a single node or a node pair, which is abstracted for simplicity by describing only the i-th node. Both datasets share a consistent structure in terms of node time-series features, OD movements, and spatial networks.

### Metadata for Epidemic-Korea dataset
```code
{
  "@type": "sc:Dataset",
  "name": "Epidemic-Korea",
  "description": "This dataset contains epidemic data for South Korea, including spatial network data, node time series features, and origin-destination movements.",
  "license": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
  "url": "https://drive.google.com/file/d/1G0P4-HdRoU6X2p18VmOaEMJVPIUEEhiE/view?usp=sharing",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "SPATIAL_NETWORK.csv",
      "name": "SPATIAL_NETWORK.csv",
      "contentUrl": "Epidemic-Korea/SPATIAL_NETWORK.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "NODE_TIME_SERIES_FEATURES.csv",
      "name": "NODE_TIME_SERIES_FEATURES.csv",
      "contentUrl": "Epidemic-Korea/NODE_TIME_SERIES_FEATURES.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "OD_MOVEMENTS.csv",
      "name": "OD_MOVEMENTS.csv",
      "contentUrl": "Epidemic-Korea/OD_MOVEMENTS.csv",
      "encodingFormat": "text/csv"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "name": "SPATIAL_NETWORK",
      "description": "This table represents the spatial network of nodes in Seoul.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "INDEX",
          "description": "The index column containing node identifiers.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "SPATIAL_NETWORK.csv" },
            "extract": {
              "column": "INDEX"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni",
          "description": "Columns representing spatial connections between all nodes and i-th node in the spatial network to describe an adjacency matrix ",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "SPATIAL_NETWORK.csv" },
            "extract": {
              "column": "Ni"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "name": "NODE_TIME_SERIES_FEATURES",
      "description": "This table contains time series features for each node over multiple timestamps.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "datetime",
          "description": "The datetime column representing timestamps (year, month, day).",
          "dataType": "sc:DateTime",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "datetime"
            }
          }
        },
        {
          "@type": "cr:Field",
          "name": "Ni_INFECTION",
          "description": "Columns representing infection cases time series for i-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "NODE_TIME_SERIES_FEATURES.csv" },
            "extract": {
              "column": "Ni_INFECTION"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "name": "OD_MOVEMENTS",
      "description": "This table represents origin-destination movements between nodes.",
      "field": [
        {
          "@type": "cr:Field",
          "name": "Ni_Nj",
          "description": "Columns representing movements from i-th node to j-th node.",
          "dataType": "sc:Integer",
          "references": {
            "fileObject": { "@id": "OD_MOVEMENTS.csv" },
            "extract": {
              "column": "Ni_Nj"
            }
          }
        }
      ]
    }
  ]
}
```

### Metadata for Epidemic-NYC dataset
```code
{
  "@type": "sc:Dataset",
  "name": "Epidemic-NYC",
  "description": "This dataset contains epidemic data for NYC, including spatial network data, node time series features, and origin-destination movements.",
  "license": "https://creativecommons.org/licenses/by-nc/4.0/",
  "url": "https://drive.google.com/file/d/1G0P4-HdRoU6X2p18VmOaEMJVPIUEEhiE/view?usp=sharing",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "SPATIAL_NETWORK.csv",
      "name": "SPATIAL_NETWORK.csv",
      "contentUrl": "Epidemic-NYC/SPATIAL_NETWORK.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "NODE_TIME_SERIES_FEATURES.csv",
      "name": "NODE_TIME_SERIES_FEATURES.csv",
      "contentUrl": "Epidemic-NYC/NODE_TIME_SERIES_FEATURES.csv",
      "encodingFormat": "text/csv"
    },
    {
      "@type": "cr:FileObject",
      "@id": "OD_MOVEMENTS.csv",
      "name": "OD_MOVEMENTS.csv",
      "contentUrl": "Epidemic-NYC/OD_MOVEMENTS.csv",
      "encodingFormat": "text/csv"
    }
  ]
}
```
