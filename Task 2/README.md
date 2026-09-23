# Customer Segmentation

This project groups customers based on two features:

- Annual Income
- Spending Score

I used **K-Means clustering** as the main method and also tested **DBSCAN** as a bonus.

## Dataset

The project uses the Mall Customers dataset.

It contains **200 customers** with information including:

- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

For the clustering part, I focused on:

- `Annual Income (k$)`
- `Spending Score (1-100)`

## Approach

The notebook follows these steps:

1. Load and inspect the dataset
2. Check the data
3. Visualize the customer distribution
4. Scale the clustering features
5. Use the Elbow Method to choose the number of clusters
6. Check the Silhouette Score
7. Apply K-Means
8. Visualize the resulting clusters
9. Compare the average spending score for each cluster
10. Test DBSCAN as a bonus
11. Summarize the results

## K-Means Result

The Elbow Method showed a clear change around **5 clusters**.

The Silhouette Score was also highest at **5 clusters**, so I used:

```text
Number of clusters = 5
```

The clusters are based on Annual Income and Spending Score.

## DBSCAN

I also tested DBSCAN to see how a density-based clustering method performs on the same data.

For this dataset, K-Means produced clearer customer segments for the two selected features.

## How to Run

1. Make sure Python is installed.
2. Install the required libraries if needed:

```bash
pip install pandas matplotlib scikit-learn
```

3. Open `Customer_Segmentation.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Make sure `Mall_Customers.csv` is in the same folder.
5. Run the notebook from top to bottom.

## Files

```text
Task 2/
├── Customer_Segmentation.ipynb
├── Mall_Customers.csv
└── README.md
```

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
