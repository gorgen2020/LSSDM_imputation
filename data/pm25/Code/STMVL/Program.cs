using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STMVL
{
    class Program
    {
        static void Main(string[] args)
        {
            Trace.TraceInformation("");
            Trace.TraceInformation("Start new iteator, at... ");
            Trace.Flush();

            //verify();
            //stmv();
            Trace.Flush();

            Console.WriteLine("finish");
            Console.ReadLine();
        }

        public static void verify()
        {
            // row means timestamp
            // column means sensors
            // pls refer to the sample data
            // we leave the missing as blank.
            // When wrapping code, we fix some small bugs, finding a better result than paper.

            string missingFile = @"SampleData/pm25_missing.txt";
            string groundFile = @"SampleData/pm25_ground.txt";
            string predictFile = "pm25_predict.txt";
            string latlngFile = @"SampleData/pm25_latlng.txt";

            int windowSize = 7;
            double alpha = 4;
            double gamma = 0.85;

            int rowCount = 8759; // count of timestamps
            int colCount = 36; // count of sensors

            bool isBlockMissing = true; //for block missing problem, if exist, set true.

            STMVL algorithm = new STMVL(missingFile, latlngFile, rowCount, colCount, alpha, gamma, windowSize, true);
            algorithm.Run(predictFile, isBlockMissing);

            Trace.TraceInformation("Evaluate... " + DateTime.Now);
            algorithm.Evaluate(groundFile);

            Console.WriteLine(DateTime.Now);
        }

        public static void stmv()
        {
            // Fill missing for whole dataset

            string missingFile = @"SampleData/pm25_missing.txt";
            string predictFile = "pm25_predict.txt";
            string latlngFile = @"SampleData/pm25_latlng.txt";

            int windowSize = 7;
            double alpha = 4;
            double gamma = 0.85;

            int rowCount = 8759; // count of timestamps
            int colCount = 36; // count of sensors

            bool isInitialize = true;  //for block missing problem

            STMVL algorithm = new STMVL(missingFile, latlngFile, rowCount, colCount, alpha, gamma, windowSize);
            algorithm.Run(predictFile, isInitialize);
           
            Console.WriteLine(DateTime.Now);
        }
    }
}
