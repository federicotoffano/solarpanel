using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using AForge.Imaging;
using AForge.Math;
using AForge;
using AForge.Math.Geometry;

namespace ecoprogetti
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            fileNameTextBox.Text = "C:\\Users\\Federico\\Desktop\\Esempi immagini\\back contact.TIF";

        }

        private void button1_Click(object sender, EventArgs e)
        {

            
            IntPtr image = CvInvoke.cvCreateImage(new System.Drawing.Size(400, 300), IplDepth.IplDepth_8U, 1);
        }

        private void loadImageButton_Click(object sender, EventArgs e)
        {
            DialogResult result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK || result == DialogResult.Yes)
            {
                fileNameTextBox.Text = openFileDialog1.FileName;
            }
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            processing();
        }

        public void processing()
        {
            //Load the image from file and resize it for display
            Image<Bgr, Byte> img =
               new Image<Bgr, byte>(fileNameTextBox.Text);

            originalImageBox.Image = img;

            //Convert the image to grayscale and filter out the noise
            UMat uimage = new UMat();
            CvInvoke.CvtColor(img, uimage, ColorConversion.Bgr2Gray);

            //uimage = findSquares(uimage);

            //use image pyr to remove noise
            //UMat pyrDown = new UMat();
            //CvInvoke.PyrDown(uimage, pyrDown);
            //CvInvoke.PyrUp(pyrDown, uimage);

            //double cannyThreshold = 100.0;
            //double cannyThresholdLinking = 20.0;
            //UMat cannyEdges = new UMat();
            //CvInvoke.Canny(uimage, uimage, cannyThreshold, cannyThresholdLinking);
            CvInvoke.GaussianBlur(uimage, uimage, new System.Drawing.Size(17, 17), 0, 0);
            CvInvoke.Laplacian(uimage, uimage, DepthType.Cv8U, 7, 1, 5);
            CvInvoke.Threshold(uimage, uimage, 254, 255, ThresholdType.Binary);
            //CvInvoke.AdaptiveThreshold(uimage, uimage, 255, AdaptiveThresholdType.GaussianC, ThresholdType.Binary,7,0);
            UMat morph = uimage.Clone();

            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(5, 5), new System.Drawing.Point(-1, -1));
            //CvInvoke.MorphologyEx(morph, morph, MorphOp.Close, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Isolated, new MCvScalar());
            //kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(3, 3), new System.Drawing.Point(-1, -1));

            CvInvoke.GaussianBlur(morph, morph, new System.Drawing.Size(17, 17), 0, 0);
            //CvInvoke.MorphologyEx(morph, morph, MorphOp.Gradient, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Replicate, new MCvScalar());
            CvInvoke.MorphologyEx(morph, morph, MorphOp.Open, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Replicate, new MCvScalar());

            CvInvoke.Threshold(morph, morph, 104, 255, ThresholdType.Binary);
            CvInvoke.GaussianBlur(morph, morph, new System.Drawing.Size(17, 17), 0, 0);

            CvInvoke.Threshold(morph, morph, 104, 255, ThresholdType.Binary);

            CvInvoke.MorphologyEx(morph, morph, MorphOp.Open, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Replicate, new MCvScalar());

            //for (int r = 1; r < 2; r++)
            //{
            //    Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new System.Drawing.Size(2, 2), new System.Drawing.Point(-1, -1));
            //    CvInvoke.MorphologyEx(morph, morph, MorphOp.Close, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            //    CvInvoke.MorphologyEx(morph, morph, MorphOp.Open, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            //}

            //double cannyThreshold = 250.0;
            //double cannyThresholdLinking = 250.0;
            //UMat cannyEdges = new UMat();
            //CvInvoke.Canny(morph, morph, cannyThreshold, cannyThresholdLinking);

            //Bitmap filt = new Bitmap( morph.Bitmap.Width, morph.Bitmap.Height);
            //Bitmap mp = new Bitmap(morph.Bitmap);
            //short[,] vse = new short[3, 3] {
            //    { 0, 0, 0 },
            //    { 0, 1, 0 },
            //    { 0, 0, 0 }
            //};
            //AForge.Imaging.Filters.HitAndMiss vFilter =
            //    new AForge.Imaging.Filters.HitAndMiss(vse);
            //Bitmap vImage = vFilter.Apply(morph.Bitmap);

            //for (int column = 0; column < vImage.Width; column++)
            //{
            //    for (int row = 0; row < vImage.Height; row++)
            //    {
            //        Color p1 = vImage.GetPixel(column, row);
            //        Color p2 = mp.GetPixel(column, row);
            //        if (p1.ToString() == p2.ToString())
            //            filt.SetPixel(column, row, Color.Black);
            //        else
            //            filt.SetPixel(column, row, p2);


            //    }
            //}

            //morph = (new Image<Bgr, Byte>(filt)).ToUMat();


            //LineSegment2D[] lines = CvInvoke.HoughLinesP(
            //   morph,
            //   1, //Distance resolution in pixel-related units
            //   Math.PI / 45.0, //Angle resolution measured in radians.
            //   0, //threshold
            //   2, //min Line width
            //   1); //gap between lines

            //Mat lineImage = new Mat(img.Size, DepthType.Cv8U, 3);
            //lineImage.SetTo(new MCvScalar(0));
            //foreach (LineSegment2D line in lines)
            //    CvInvoke.Line(lineImage, line.P1, line.P2, new Bgr(Color.Green).MCvScalar, 2);


            editedImageBox.Image = morph;
        }

        private void editedImageBox_Click(object sender, EventArgs e)
        {
        }

        private UMat findSquares(UMat img)
        {
            // Open your image
            string path = "test.png";
            Bitmap image = img.Bitmap; //(Bitmap)Bitmap.FromFile(path);

            // locating objects
            BlobCounter blobCounter = new BlobCounter();

            blobCounter.FilterBlobs = false;
            blobCounter.MinHeight = 5;
            blobCounter.MinWidth = 5;

            blobCounter.ProcessImage(image);
            Blob[] blobs = blobCounter.GetObjectsInformation();

            // check for rectangles
            SimpleShapeChecker shapeChecker = new SimpleShapeChecker();


            foreach (var blob in blobs)
            {
                List<IntPoint> edgePoints = blobCounter.GetBlobsEdgePoints(blob);
                List<IntPoint> cornerPoints;

                // use the shape checker to extract the corner points
                if (shapeChecker.IsQuadrilateral(edgePoints, out cornerPoints))
                {
                    // only do things if the corners form a rectangle
                    if (shapeChecker.CheckPolygonSubType(cornerPoints) == PolygonSubType.Rectangle)
                    {
                        // here i use the graphics class to draw an overlay, but you
                        // could also just use the cornerPoints list to calculate your
                        // x, y, width, height values.
                        List<AForge.Point> Points = new List<AForge.Point>();
                        foreach (var point in cornerPoints)
                        {
                            Points.Add(new AForge.Point(point.X, point.Y));
                        }

                        PointF[] P = new PointF[Points.Count];
                        for (int i =0; i < P.Length; i++)
                        {
                            P[i] = new PointF(Points[i].X, Points[i].Y);
                        }
                        Bitmap newBitmap = new Bitmap(image.Width, image.Height);
                        Graphics g = Graphics.FromImage(newBitmap);
                        g.DrawImage(image, 0, 0);
                        //Graphics g = Graphics.FromImage(image);
                        g.DrawPolygon(new Pen(Color.Red, 5.0f), P);

                        image = newBitmap;

                        //image.Save("result.png");
                    }
                }
            }
            return (new Image<Bgr, Byte>(image)).ToUMat();
        }
    }
}
