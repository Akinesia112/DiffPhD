//#pragma once
//
//#include <vector>
//#include <algorithm>
//
//#include <igl/massmatrix.h>
//#include <urdf/model.h>
//#include <string>
//#include <iostream>
//#include <exception>
//#include <tinyxml/txml.h>
//
//namespace RealSim
//{
//    class Surdfloader
//    {
//        public:
//            Surdfloader();
//            ~Surdfloader();
//
//            void ImportUrdf(string& urdf_file);
//            void CheckValid();
//            void ExportVisuals();
//            void SetupSimulatedGeo();
//
//        private:
//            string urdf_string;
//            std::shared_ptr<Sgeometry> _sgeometry;
//    };
//
//    class Salembicloader
//    {
//        public:
//            Salembicloader();
//            ~Salembicloader();
//
//            void ImportAlembic(string& abc_file);
//            void CheckValid();
//            void FrameStatusInterpolation();
//            void SetupSimulatedStatusFrame();
//
//        private:
//            string abc_string;
//            std::shared_ptr<Sgeometry> _sgeometry;
//    };
//
//    class Sobjloader
//    {
//        public:
//            Sobjloader();
//            ~Sobjloader();
//
//            void Importobj(string& obj_file);
//            void CheckValid();
//            void SetupSimulatedGeo();
//
//        private:
//            string obj_string;
//            std::shared_ptr<Sgeometry> _sgeometry;
//    };
//}