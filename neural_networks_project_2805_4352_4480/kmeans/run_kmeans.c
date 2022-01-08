#include "kmeans.h"

int main()
{
    float error = 0.0;
    float min_error = kmeans("dataset2.txt", 300);
    float** labeled_data = read_file("labeled_data.txt", LABELED_SET);
    float** kmeans_clusters = read_file("kmeans_clusters.txt", LABELED_SET);
    for (size_t i = 0; i < 20; i++)
    {
        error = kmeans("dataset2.txt", 300);
        if (error < min_error)
        {   
            // Saving best run.
            min_error = error;
            labeled_data = read_file("labeled_data.txt", LABELED_SET);
            kmeans_clusters = read_file("kmeans_clusters.txt", LABELED_SET);
        }
    }
    int len = get_file_len("labeled_data.txt");

    // Saving result to different files.
    write_labeled_dataset_to_file("labeled_data_final.txt", labeled_data, len);
    write_labeled_dataset_to_file("kmeans_clusters_final.txt", kmeans_clusters, NUM_OF_CLUSTERS);
    printf("Min error: %f for %d clusters.\n", min_error, NUM_OF_CLUSTERS);

    return 0;
}
