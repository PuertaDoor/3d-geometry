// Imagine++ project
// Project:  Panorama
// Author:   Porte Léo
// Date:     2023/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;


//add a new point to the array of the related window and highlights it
void new_point_win(Window ctxt_win,vector<IntPoint2>& w_pts,IntPoint2 new_point){
    w_pts.push_back(new_point);
    setActiveWindow(ctxt_win);
    fillCircle(new_point, 1, RED);
}


// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2, vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    Window ctxt_win;
    IntPoint2 new_point;

    int subWin;
    while(true) {
        //right click 3;
        if(anyGetMouse(new_point, ctxt_win, subWin) == 3){
            return;
        }


        if (ctxt_win==w1){
            //Add a point to window 1
            new_point_win(ctxt_win,pts1, new_point);
        }else{
            //Add a point to window 2
            new_point_win(ctxt_win,pts2, new_point);
        }

    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------
    for(size_t i=0; i<n*2; i+=2) {
        int idx_i=(int) (i/2.0);
        double x1 = pts1[idx_i].x();
        double y1 = pts1[idx_i].y();
        double x2 = pts2[idx_i].x();
        double y2 = pts2[idx_i].y();


        A(i,0) = x1;
        A(i,1) = y1;
        A(i,2) = 1;
        A(i,3) = 0;
        A(i,4) = 0;
        A(i,5) = 0;
        A(i,6) = -x2*x1;
        A(i,7) = -x2*y1;


        A(i+1,0) = 0;
        A(i+1,1) = 0;
        A(i+1,2) = 0;
        A(i+1,3) = x1;
        A(i+1,4) = y1;
        A(i+1,5) = 1;
        A(i+1,6) = -y2*x1;
        A(i+1,7) = -y2*y1;

        //  AX = B
        B[i] = x2;
        B[i+1] = y2;
    }

    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;
}

// Function to check if coordinates are within the bounds of the image
bool isInBounds(const Vector<float>& coords, const Image<Color,2>& image) {
    return (coords[0] >= 0 && coords[1] >= 0 && coords[0] < image.width() && coords[1] < image.height());
}

// Function to apply homographic transformation
Vector<float> applyHomography(const Vector<float>& coords, const Matrix<float>& H) {
    Vector<float> transformed_coords = H * coords;
    return transformed_coords / transformed_coords[2];
}

int min(int x1, int x2){
    if (x1 < x2){
        return x1;
    }
    else {
        return x2;
    }
}

void createPanorama(const Image<Color,2>& img1, const Image<Color,2>& img2,
                    Matrix<float> H, Image<Color>& result, float x0, float y0) {

    H = inverse(H);  // Inverse of matrix H

    // Determine the overlap zone
    float overlapStart = max(0.0f, x0);
    float overlapEnd = min(img1.width(), img2.width() + x0); // Assuming both images have the same width

    // Loop through the height and width of the final image
    for(int row = 0; row < result.height(); row++)
        for(int col = 0; col < result.width(); col++) {
            Vector<float> coords(3);
            coords[0] = col + x0;
            coords[1] = row + y0;
            coords[2] = 1;

            bool withinImg2 = isInBounds(coords, img2);
            Color colorImg2;
            if(withinImg2)
                colorImg2 = img2.interpolate(coords[0], coords[1]);

            coords = applyHomography(coords, H);

            if(isInBounds(coords, img1)) {
                Color colorImg1 = img1.interpolate(coords[0], coords[1]);

                if(withinImg2 && col + x0 >= overlapStart && col + x0 <= overlapEnd) {
                    // Weighted average
                    float weightI1 = (overlapEnd - (col + x0)) / (overlapEnd - overlapStart);
                    float weightI2 = 1.0f - weightI1;

                    result(col, row).r() = weightI1 * colorImg1.r() + weightI2 * colorImg2.r();
                    result(col, row).g() = weightI1 * colorImg1.g() + weightI2 * colorImg2.g();
                    result(col, row).b() = weightI1 * colorImg1.b() + weightI2 * colorImg2.b();
                } else if (withinImg2) {
                    result(col, row) = colorImg2;
                } else {
                    result(col, row) = colorImg1;
                }
            } else if (withinImg2) {
                result(col, row) = colorImg2;
            }
        }
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    createPanorama(I1, I2, H, I, x0, y0);


    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);
    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
