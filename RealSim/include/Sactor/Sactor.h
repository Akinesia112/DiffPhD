//# pragma once
//
///*
//The basic Type for Sactor is an object, which could integrate with many kinds of simulated types,
//such as rigid, dynamics and fluid, each simulated parts act as "component".
//*/
//
//#include <iostream>
//#include <vector>
//#include <memory>
//
//// header for other ".h" file
//#include "Scomponent/integrator/SBaseIntegrator.h"
//
//namespace RealSim
//{
//    using namespace std;
//
//    class Sactor
//    {
//        public:
//            Sactor();
//            Sactor(string sactor_name);
//
//            void Initialize();
//            void Start();
//            void Update();
//            void OnDestroy();
//
//            void AddScomponent(shared_ptr<Scomponent> component);
//            void AddScomponents(const initializer_list<shared_ptr<Scomponent>>& newcomponents);
//
//            template <typename T>
//            void DestroyScomponent();
//
//        private:
//            vector<shared_ptr<Scomponent>> Scomponents;
//            string Sactor_name;
//
//    };
//}