```mermaid
flowchart TD
    A[Входные данные] --> B[Изображение глазного дна\n224x224x3]
    A --> C[Клинические данные:\nGESTATIONAL_AGE, BIRTH_WEIGHT,\nPOSTCONCEPTUAL_AGE, SEX]
    
    B --> D[ResNet50\n(заморожено 90% слоев)]
    D -->|2048-D\nвизуальные фичи| F[Объединение признаков]
    
    C --> E[Feature Processor]
    E -->|64-D\nклинические фичи| F
    
    F --> G[Классификатор]
    G --> H[256-D\nскрытый слой\nDropout 0.6]
    H --> I[2-D\nвыход: 0/1]
    
    subgraph CNN[Визуальный путь]
        D -->|Слои ResNet| D1[Conv1 7x7]
        D1 --> D2[MaxPool]
        D2 --> D3[Layer1-3\nзаморожены]
        D3 --> D4[Layer4\nобучается]
        D4 --> D5[AvgPool]
    end
    
    subgraph Clinical[Клинический путь]
        E --> E1[Linear 4→32\nDropout 0.4]
        E1 --> E2[Linear 32→64]
    end
    
    subgraph Classifier[Классификатор]
        G --> G1[Linear 2048+64→256]
        G1 --> G2[ReLU]
        G2 --> G3[Dropout 0.6]
        G3 --> G4[Linear 256→2]
    end

```