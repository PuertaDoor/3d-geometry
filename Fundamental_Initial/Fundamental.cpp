// Imagine++ project
// Project:  Fundamental
// Author:   Léo Porte

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Normalize point matches and return normalization matrices
pair<FMatrix<float,3,3>, FMatrix<float,3,3>> normalize(vector<Match>& matches) {
    // Define the normalization matrix N
    FMatrix<float,3,3> N;
    N.fill(0.0f);
    N(0,0) = 1e-3;
    N(1,1) = 1e-3;
    N(2,2) = 1.0f;

    // Normalize the points
    for (Match& m : matches) {
        FVector<float,3> p1(m.x1, m.y1, 1.0f);
        FVector<float,3> p2(m.x2, m.y2, 1.0f);

        p1 = N * p1;
        p2 = N * p2;

        m.x1 = p1[0];
        m.y1 = p1[1];
        m.x2 = p2[0];
        m.y2 = p2[1];
    }

    return {N, N}; // Return normalization matrices for both sets of points
}

// Dot product between two vectors
float dot(FVector<float, 3> v1, FVector<float, 3> v2){
    float result = 0.0f;
    for (int i = 0; i < v1.size(); i++){
        result += v1[i] * v2[i];
    }
    return result;
}

/*
In vanilla epipolar distance, the inlier determination is based on the epipolar constraint:

x′@ F @ x = 0

Where:

    x is a point in the first image.
    x′ is the corresponding point in the second image.
    F is the fundamental matrix.

The code checks the value of x′Fx (which is stored in the error variable in the computeF implementation)
and considers a match as an inlier if the absolute value of this error is below a threshold (distMax).
This is essentially measuring the distance of the point x′ to the epipolar line in the second image defined by x and F.
However, it does not consider the error in the first image, i.e., the distance of the point x to the epipolar line in the first image defined by x′ and F.

To solve this problem, the symmetric epipolar distance is implemented just below this comment.

The symmetric epipolar distance is a measure that takes into account the geometric error between a point and its corresponding epipolar line in both images.

The symmetric epipolar distance is beneficial because:

    Balanced Error Measurement: It provides a balanced measure of the error in both images. This is especially useful when one image might have more noise or distortion than the other.
    Robustness: By considering errors in both images, it can offer a more robust measure for inlier determination, especially in the presence of noise or outliers.
    Geometric Interpretation: The symmetric epipolar distance has a clear geometric interpretation, making it intuitive to understand and use.

In summary, we can use a one-sided epipolar distance, but the symmetric epipolar distance considers the error in both images, making it a more balanced and potentially more robust measure.

It is defined by:

    d(p,l')^2 + d(p',l)^2

Where:

- p is a point in the first image.
- p' is its corresponding point in the second image.
- l' is the epipolar line in the second image corresponding to point p.
- l is the epipolar line in the first image corresponding to point p'
- d(p,l') is the distance between point p and line l'.
- d(p',l) is the distance between point p' and line l.

Given the fundamental matrix F, the epipolar lines l and l' can be computed as:

    l = transpose(F) @ p'
    l' = F @ p

The distance between a point p = transpose(x,y,1) and a line l = transpose(a,b,c) is given by:

    d(p,l) = |ax+by+c| / sqrt(a^2 + b^2)
*/

float symmetricEpipolarDistance(const FVector<float,3>& p, const FVector<float,3>& p_prime, const FMatrix<float,3,3>& F) {
    // Compute epipolar lines
    FVector<float,3> l_prime = F * p;
    FVector<float,3> l = transpose(F) * p_prime;

    // Compute distances
    float dist_p_to_l_prime = abs(dot(p, l_prime)) / sqrt(l_prime[0]*l_prime[0] + l_prime[1]*l_prime[1]);
    float dist_p_prime_to_l = abs(dot(p_prime, l)) / sqrt(l[0]*l[0] + l[1]*l[1]);

    // Return symmetric epipolar distance
    return dist_p_to_l_prime * dist_p_to_l_prime + dist_p_prime_to_l * dist_p_prime_to_l;
}


// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches, float inlierThreshold = 1.5f, int maxIterations = 100000, bool symmetric = false) {
    const float distMax = inlierThreshold; // Pixel error for inlier/outlier discrimination
    int Niter = maxIterations; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;

    // Normalize the points
    auto [T1, T2] = normalize(matches);

    for (int iter = 0; iter < Niter; iter++) {
        vector<int> inliers;
        // Randomly select 8 matches
        vector<Match> randomMatches;
        for (int i = 0; i < 8; i++) {
            int idx = rand() % matches.size();
            randomMatches.push_back(matches[idx]);
        }

        // Construct matrix A
        Matrix<float> A(8,9);
        for (int i = 0; i < 8; i++) {
            Match m = randomMatches[i];
            A(i, 0) = m.x1 * m.x2;
            A(i, 1) = m.x1 * m.y2;
            A(i, 2) = m.x1;
            A(i, 3) = m.y1 * m.x2;
            A(i, 4) = m.y1 * m.y2;
            A(i, 5) = m.y1;
            A(i, 6) = m.x2;
            A(i, 7) = m.y2;
            A(i, 8) = 1.0f;
        }

        // Compute SVD of A
        Matrix<float> U, V;
        Vector<float> S;

        svd(A, U, S, V);

        // Construct F from last column of V
        Matrix<float> F(3,3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F(i, j) = V(V.nrow(), 3 * i + j);
            }
        }

        // Enforce rank 2 constraint on F
        svd(F, U, S, V);
        S[2] = 0.0f; // Set the smallest singular value to 0
        F = U * Diagonal(S) * transpose(V);

        // convert Matrix to FMatrix
        FMatrix<float, 3,3> FF;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                FF(i, j) = F(i,j);
            }
        }

        // Denormalize F
        FF = transpose(T2) * FF * T1;

        // Count inliers
        inliers.clear();
        for (int i = 0; i < matches.size(); i++) {
            Match m = matches[i];
            if (symmetric == false){
                FMatrix<float, 1, 3> p1;
                p1(0,0) = m.x1;
                p1(0,1) = m.y1;
                p1(0,2) = 1.0f;
                FMatrix<float, 3, 1> p2;
                p1(0,0) = m.x2;
                p1(1,0) = m.y2;
                p1(2,0) = 1.0f;
                FMatrix<float, 1, 1> temp = p1 * FF * p2;
                float error = temp(0,0);
                if (abs(error) < distMax) {
                    inliers.push_back(i);
                }
            }
            else {
                FVector<float, 3> p1(m.x1, m.y1, 1.0f);
                FVector<float, 3> p2(m.x2, m.y2, 1.0f);

                float distance = symmetricEpipolarDistance(p1, p2, FF);
                if (distance < distMax) {
                    inliers.push_back(i);
                }
            }
        }

        // Update bestF and bestInliers if current inliers are greater
        if (inliers.size() > bestInliers.size()) {
            bestF = FF;
            bestInliers = inliers;
            // Update Niter based on inlier ratio
            float w = (float)inliers.size() / matches.size();
            float pNoOutliers = 1.0f - pow(w, 8);
            pNoOutliers = max(1e-9f, pNoOutliers); // Avoid division by zero
            pNoOutliers = min(1.0f - 1e-9f, pNoOutliers); // Avoid taking log(0)
            Niter = log(BETA) / log(pNoOutliers);
        }
    }

    // Refine F using best inliers
    Matrix<float> A(matches.size(), 9);
    for (size_t i = 0; i < matches.size(); i++) {
        Match m = matches[i];
        A(i, 0) = m.x1 * m.x2;
        A(i, 1) = m.x1 * m.y2;
        A(i, 2) = m.x1;
        A(i, 3) = m.y1 * m.x2;
        A(i, 4) = m.y1 * m.y2;
        A(i, 5) = m.y1;
        A(i, 6) = m.x2;
        A(i, 7) = m.y2;
        A(i, 8) = 1.0f;
    }

    // Compute SVD of A
    Matrix<float> U, V;
    Vector<float> S;
    svd(A, U, S, V);

    // Construct F from last column of V
    Matrix<float> F(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            F(i, j) = V(V.nrow(), 3 * i + j);
        }
    }

    // Enforce rank 2 constraint on F
    svd(F, U, S, V);
    S[2] = 0.0f; // Set the smallest singular value to 0
    F = U * Diagonal(S) * transpose(V);

    // convert Matrix to FMatrix
    FMatrix<float, 3,3> FF;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            FF(i, j) = F(i,j);
        }
    }

    // Denormalize F
    bestF = transpose(T2) * FF * T1;

    // Update matches with inliers only
    vector<Match> all = matches;
    matches.clear();
    for (size_t i = 0; i < bestInliers.size(); i++) {
        matches.push_back(all[bestInliers[i]]);
    }

    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    int w = I1.width();
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;

        // Check if the click is on the left or right image
        bool isLeftImage = (x < w);

        FVector<float,3> point(x, y, 1.0f);
        FVector<float,3> line;

        if(isLeftImage) {
            // If click is on the left image, compute epipolar line in right image
            line = F * point;
        } else {
            // If click is on the right image, compute epipolar line in left image
            line = transpose(F) * point;
        }

        // Compute the two endpoints of the epipolar line to draw it
        // We use the intersections of the line with the image borders
        FVector<float,2> pt1, pt2;
        if(abs(line[0]) > abs(line[1])) {
            // Line is more horizontal, use top and bottom borders
            pt1 = FVector<float,2>(-line[2]/line[0], 0);
            pt2 = FVector<float,2>(-(line[2]+line[1]*I1.height())/line[0], I1.height());
        } else {
            // Line is more vertical, use left and right borders
            pt1 = FVector<float,2>(0, -line[2]/line[1]);
            pt2 = FVector<float,2>(I1.width(), -(line[2]+line[0]*I1.width())/line[1]);
        }

        // Adjust for right image coordinates if necessary
        if(!isLeftImage) {
            pt1[0] += w;
            pt2[0] += w;
        }

        // Draw the clicked point and the epipolar line
        Color c(rand()%256,rand()%256,rand()%256); // Random color for visualization
        if(isLeftImage) {
            fillCircle(x, y, 3, c); // Point in left image
            drawLine(pt1[0]+w, pt1[1], pt2[0]+w, pt2[1], c); // Line in right image
        } else {
            fillCircle(x, y, 3, c); // Point in right image
            drawLine(pt1[0], pt1[1], pt2[0], pt2[1], c); // Line in left image
        }
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches, 1.5f, 10000, true);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
