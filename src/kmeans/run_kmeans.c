#include "kmeans.h"

int main()
{
    float error = 0.0;
    float min_error = kmeans("../../data/dataset2.txt", 300);
    float** labeled_data = read_file("../../out/labeled_data.txt", LABELED_SET);
    float** kmeans_clusters = read_file("../../out/kmeans_clusters.txt", LABELED_SET);
    for (size_t i = 0; i < 20; i++)
    {
        error = kmeans("../../data/dataset2.txt", 300);
        // printf("Iteration %ld: Intra cluster variance: %.2f\n", i, error);
        if (error < min_error)
        {   
            min_error = error;
            labeled_data = read_file("../../out/labeled_data.txt", LABELED_SET);
            kmeans_clusters = read_file("../../out/kmeans_clusters.txt", LABELED_SET);
        }
    }
    int len = get_file_len("../../out/labeled_data.txt");
    write_labeled_dataset_to_file("../../out/SEL_labeled_data.txt", labeled_data, len);
    write_labeled_dataset_to_file("../../out/SEL_kmeans_clusters.txt", kmeans_clusters, NUM_OF_CLUSTERS);
    printf("Min error: %f for %d clusters.\n", min_error, NUM_OF_CLUSTERS);

    return 0;
}
