#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/readMSH.h>

#include "Sgeometry/STetrahedraMesh.h"
#include "Scomponent/tools/bbox/AABB.h"
#include "Scomponent/integrator/localglobal/LocalGlobalSolver.h"
#include "Scomponent/integrator/localglobal/energy/hardconstraint/DynamicPinEnergy.h"
#include "types.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;
using namespace RealSim;

struct SimulationData
{
    MatXR positions;  //vertices position
    std::vector<MatXR> meshV;  //vertices position for each mesh
    std::vector<MatXI> meshT;  //tetrahedra
    std::vector<MatXI> meshF;  //triangles
    std::vector<MatXI> meshE;  //edges

    std::vector<VecXI> mesh_vertices;
    int total_vertices = 0;

    std::vector<Vec3R> translations;
    std::vector<Vec3R> rotations;
    std::vector<Vec3R> scales;
    std::vector<Vec3R> centers;

    unsigned int num_meshes() const {return mesh_vertices.size();}

    const std::vector<VecXI> & getVertices() {return mesh_vertices;}

    void clear()
    {
        meshV.clear();
        meshT.clear();
        meshF.clear();
        meshE.clear();
        mesh_vertices.clear();
        translations.clear();
        rotations.clear();
        scales.clear();
        centers.clear();
        total_vertices = 0;
    }

    void addVertices(MatXR& pos,
                     const Vec3R& _translation = Vec3R(0.0, 0.0, 0.0),
                     const Vec3R& _rotation = Vec3R(0.0, 0.0, 0.0),
                     const Vec3R& _scale3D = Vec3R(1.0, 1.0, 1.0),
                     const Vec3R& _center = Vec3R(0.0, 0.0, 0.0))
    {
        translations.push_back(_translation);
        rotations.push_back(_rotation);
        scales.push_back(_scale3D);
        centers.push_back(_center);

        //// apply scales
        if(_scale3D != Vec3R(1.0, 1.0, 1.0))
        {
            Mat3R scale_matrix = _scale3D.asDiagonal();

            for (int i = 0; i < pos.rows(); i++) {
                Vec3R orient = (Vec3R)pos.row(i) - _center;
                Vec3R rotated_orient = scale_matrix * orient;
                pos.row(i) = _center + rotated_orient;
            }
        }

        //// apply rotations
        if(_rotation != Vec3R(0.0, 0.0, 0.0))
        {
            Mat3R rotation_matrix;

            real c_alpha = cos(_rotation[0] * M_PI / 180.0);
            real s_alpha = sin(_rotation[0] * M_PI / 180.0);
            real c_beta = cos(_rotation[1] * M_PI / 180.0);
            real s_beta = sin(_rotation[1] * M_PI / 180.0);
            real c_gamma = cos(_rotation[2] * M_PI / 180.0);
            real s_gamma = sin(_rotation[2] * M_PI / 180.0);

            rotation_matrix(0, 0) = c_gamma * c_beta;
            rotation_matrix(0, 1) = c_gamma * s_beta * s_alpha - s_gamma * c_alpha;
            rotation_matrix(0, 2) = c_gamma * s_beta * c_alpha + s_gamma * s_alpha;

            rotation_matrix(1, 0) = s_gamma * c_beta;
            rotation_matrix(1, 1) = s_gamma * s_beta * s_alpha + c_gamma * c_alpha;
            rotation_matrix(1, 2) = s_gamma * s_beta * c_alpha - c_gamma * s_alpha;

            rotation_matrix(2, 0) = - s_beta;
            rotation_matrix(2, 1) = c_beta * s_alpha;
            rotation_matrix(2, 2) = c_beta * c_alpha;

            for (int i = 0; i < pos.rows(); i++) {
                Vec3R orient = (Vec3R)pos.row(i) - _center;
                Vec3R rotated_orient = rotation_matrix * orient;
                pos.row(i) = _center + rotated_orient;
            }
        }

        //// apply translations
        for (int i = 0; i < pos.rows(); i++) {
            pos.row(i) += _translation;
        }

        meshV.push_back(pos);
        auto copy_pos = positions;
        positions.resize(copy_pos.rows()+pos.rows(), 3);
        positions<<copy_pos, pos;

        VecXI vertices;
        vertices.resize(pos.rows());
        for(int j = 0; j < pos.rows(); j++) vertices[j] = (total_vertices + j);
        mesh_vertices.push_back(vertices);
        total_vertices += pos.rows();
    }

    void addTetrahedra(MatXI& tetras)
    {
        tetras.array() += total_vertices;
        meshT.push_back(tetras);
    }

    void addTriangles(MatXI& triangles)
    {
        triangles.array() += total_vertices;
        meshF.push_back(triangles);
    }

    void addEdges(MatXI& edges)
    {
        edges.array() += total_vertices;
        meshE.push_back(edges);
    }
};

void addDynamicVisualSpot(std::vector<polyscope::SurfaceMesh *> & Spots, SimulationData & Data, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Meshes, real transparency = 1.0, real edgeWidth = 0.0)
{
    for(const auto& mesh: Meshes)
    {
        if(mesh->getSurfaceTriangles() != nullptr)
        {
            Spots.push_back(polyscope::registerSurfaceMesh("Dynamic Mesh Surface" + std::to_string(Spots.size()), Data.positions, *mesh->getSurfaceTriangles()));
            Spots.back()->setEdgeWidth(edgeWidth);
            Spots.back()->setTransparency(transparency);
        }
    }
}

void addStaticVisualSpot(std::vector<polyscope::SurfaceMesh *> & Spots, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Meshes, real transparency = 1.0, real edgeWidth = 0.0)
{
    for(const auto& mesh: Meshes)
    {
        if(mesh->getSurfaceTriangles() != nullptr)
        {
            Spots.push_back(polyscope::registerSurfaceMesh("Static Mesh Surface" + std::to_string(Spots.size()), *mesh->getPositions(), *mesh->getSurfaceTriangles()));
            Spots.back()->setEdgeWidth(edgeWidth);
            Spots.back()->setTransparency(transparency);
        }
    }
}

int InitDynamics(SimulationData& Data, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Meshes, std::vector<json> &objectsconfig)
{
    Data.clear();
    Meshes.clear();

    for(auto config: objectsconfig)
    {
        std::string filename = std::string(PROJECT_SOURCE_DIR) + std::string(config["mesh"]);
        std::string element_type = std::string(config["element_type"]);

        Vec3R translation = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["trans"])).data(), 3);
        Vec3R rotation = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["rotation"])).data(), 3);
        Vec3R scale = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["scale"])).data(), 3);
        Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["center"])).data(), 3);

        if(element_type == "TRIANGLE")
        {
            MatXR meshV;
            MatXI meshF;

            igl::readOBJ(filename, meshV, meshF);

            Data.addTriangles(meshF);
            Data.addVertices(meshV, translation, rotation, scale, center);

            // generate structure for storing meshes
            auto triangle_mesh = std::make_shared<RealSim::geometry::STriangleMesh>(true, Data.getVertices().back(), Data.positions, meshF);
            triangle_mesh->init();
            Meshes.push_back(triangle_mesh);

            std::cout<<GREEN<<"adding dynamic triangle mesh, vertices = "<<BLUE<<meshV.rows()<<GREEN<<", triangles = "<<BLUE<<meshF.rows()<<RESET<<std::endl;
        }
        else if(element_type == "TETRAHEDRON")
        {
            MatXR meshV;
            MatXI meshT;
            MatXI meshF;

            igl::readMESH(filename, meshV, meshT, meshF);

            Data.addTetrahedra(meshT);
            Data.addVertices(meshV, translation, rotation, scale, center);

            // generate structure for storing meshes
            auto tetra_mesh = std::make_shared<RealSim::geometry::STetrahedraMesh>(true, Data.getVertices().back(), Data.positions, meshT);
            tetra_mesh->init();
            Meshes.push_back(tetra_mesh);

            std::cout<<GREEN<<"adding dynamic tetrahedron mesh, vertices = "<<BLUE<<meshV.rows()<<GREEN<<", triangles = "<<BLUE<<meshF.rows()<<GREEN<<", tetrahedra = "<<BLUE<<meshT.rows()<<RESET<<std::endl;
        }
    }

    return 0;
}

int InitObstacles(std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Obstacles, std::vector<json> &obstacleconfig)
{
    for(auto config: obstacleconfig)
    {
        std::string filename = std::string(PROJECT_SOURCE_DIR) + std::string(config["mesh"]);
        std::string element_type = std::string(config["element_type"]);

        Vec3R translation = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["trans"])).data(), 3);
        Vec3R rotation = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["rotation"])).data(), 3);
        Vec3R scale = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["scale"])).data(), 3);
        Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["center"])).data(), 3);

        if(element_type == "TRIANGLE")
        {
            SimulationData Data;

            MatXR meshV;
            MatXI meshF;

            igl::readOBJ(filename, meshV, meshF);

            Data.addTriangles(meshF);
            Data.addVertices(meshV, translation, rotation, scale, center);

            // generate structure for storing meshes
            auto triangle_mesh = std::make_shared<RealSim::geometry::STriangleMesh>(false, Data.getVertices().back(), Data.positions, meshF);
            triangle_mesh->init();
            Obstacles.push_back(triangle_mesh);

            std::cout<<GREEN<<"adding obstacle with triangle mesh, vertices = "<<BLUE<<meshV.rows()<<GREEN<<", triangles = "<<BLUE<<meshF.rows()<<RESET<<std::endl;
        }
        else if(element_type == "TETRAHEDRON")
        {
            SimulationData Data;

            MatXR meshV;
            MatXI meshT;
            MatXI meshF;

            igl::readMESH(filename, meshV, meshT, meshF);

            Data.addTetrahedra(meshT);
            Data.addVertices(meshV, translation, rotation, scale, center);

            // generate structure for storing meshes
            auto tetra_mesh = std::make_shared<RealSim::geometry::STetrahedraMesh>(false, Data.getVertices().back(), Data.positions, meshT);
            tetra_mesh->init();
            Obstacles.push_back(tetra_mesh);

            std::cout<<GREEN<<"adding obstacle with tetrahedron mesh, vertices = "<<BLUE<<meshV.rows()<<GREEN<<", triangles = "<<BLUE<<meshF.rows()<<GREEN<<", tetrahedra = "<<BLUE<<meshT.rows()<<RESET<<std::endl;
        }
    }

    return 0;
}

Vec3R getRotation(const Vec3R& toDirection, const Vec3R& fromDirection)
{
    Vec3R from = fromDirection.normalized();
    Vec3R to = toDirection.normalized();

    Eigen::Vector3d fromR = from.cast<double>();
    Eigen::Vector3d toR = to.cast<double>();

    // Compute the rotation matrix
    Eigen::Matrix3d rotationMatrix;
    rotationMatrix = Eigen::AngleAxisd(acos(fromR.dot(toR)), (fromR.cross(toR)).normalized());

    // Extract Euler angles from rotation matrix
    real roll, pitch, yaw;
    roll = atan2(rotationMatrix(2, 1), rotationMatrix(2, 2));
    pitch = atan2(-rotationMatrix(2, 0), sqrt(rotationMatrix(2, 1)*rotationMatrix(2, 1) + rotationMatrix(2, 2)*rotationMatrix(2, 2)));
    yaw = atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));

    return Vec3R(roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
}

void LoadPlaneMesh(std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & AnalyticalObstacle, const Vec3R& translation, const Vec3R& rotation, const Vec3R& scale = Vec3R(1,1,1))
{
    std::string filename = std::string(PROJECT_SOURCE_DIR) + "/resources/mesh/surface/plane.obj";

    MatXR meshV;
    MatXI meshF;
    igl::readOBJ(filename, meshV, meshF);

    SimulationData planeData;
    planeData.addVertices(meshV, translation, rotation, scale, Vec3R(0.0, 0.0, 0.0));

    auto triangle_mesh = std::make_shared<RealSim::geometry::STriangleMesh>(false, planeData.getVertices().back(), planeData.positions, meshF);
    AnalyticalObstacle.push_back(triangle_mesh);
}

void LoadCylinderMesh(std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & AnalyticalObstacle, const Vec3R& translation, const Vec3R& rotation, const Vec3R& scale = Vec3R(1,1,1))
{
    std::string filename = std::string(PROJECT_SOURCE_DIR) + "/resources/mesh/surface/cylinder.obj";

    MatXR meshV;
    MatXI meshF;
    igl::readOBJ(filename, meshV, meshF);

    SimulationData cylinderData;
    cylinderData.addVertices(meshV, translation, rotation, scale, Vec3R(0.0, 0.0, 0.0));

    auto triangle_mesh = std::make_shared<RealSim::geometry::STriangleMesh>(false, cylinderData.getVertices().back(), cylinderData.positions, meshF);
    AnalyticalObstacle.push_back(triangle_mesh);
}

void LoadBoxMesh(std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & AnalyticalObstacle, const Vec3R& translation, const Vec3R& rotation, const Vec3R& scale)
{
    std::string filename = std::string(PROJECT_SOURCE_DIR) + "/resources/mesh/surface/cube.obj";

    MatXR meshV;
    MatXI meshF;
    igl::readOBJ(filename, meshV, meshF);

    SimulationData boxData;
    boxData.addVertices(meshV, translation, rotation, scale, Vec3R(0.0, 0.0, 0.0));

    auto triangle_mesh = std::make_shared<RealSim::geometry::STriangleMesh>(false, boxData.getVertices().back(), boxData.positions, meshF);
    AnalyticalObstacle.push_back(triangle_mesh);
}

void LoadSphereMesh(std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & AnalyticalObstacle, const Vec3R& translation, real scale)
{
    std::string filename = std::string(PROJECT_SOURCE_DIR) + "/resources/mesh/surface/sphere.obj";

    MatXR meshV;
    MatXI meshF;
    igl::readOBJ(filename, meshV, meshF);

    SimulationData sphereData;
    sphereData.addVertices(meshV, translation, Vec3R(0.0, 0.0, 0.0), Vec3R(scale, scale, scale), Vec3R(0.0, 0.0, 0.0));

    auto triangle_mesh = std::make_shared<RealSim::geometry::STriangleMesh>(false, sphereData.getVertices().back(), sphereData.positions, meshF);
    AnalyticalObstacle.push_back(triangle_mesh);
}

int InitCollision(const std::shared_ptr<RealSim::collisiondetection::CollisionHandler>& Collision, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Meshes, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Obstacles, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & AnalyticalObstacles, json & config)
{
    if(config["genericDCD"] != nullptr)
    {
        real ms = (config["genericDCD"]["miniseparation"] != nullptr)? real(config["genericDCD"]["miniseparation"]) : 0.0;
        real alarm = (config["genericDCD"]["alarm"] != nullptr)? real(config["genericDCD"]["alarm"]) : ms;
        real muVF = (config["genericDCD"]["muVF"] != nullptr)? real(config["genericDCD"]["muVF"]) : 0.0;
        real muEE = (config["genericDCD"]["muEE"] != nullptr)? real(config["genericDCD"]["muEE"]) : 0.0;
        real muFV = (config["genericDCD"]["muFV"] != nullptr)? real(config["genericDCD"]["muFV"]) : 0.0;
        bool VF = (config["genericDCD"]["VF"] != nullptr)? bool(config["genericDCD"]["VF"]) : true;
        bool EE = (config["genericDCD"]["EE"] != nullptr)? bool(config["genericDCD"]["EE"]) : true;
        bool FV = (config["genericDCD"]["VF"] != nullptr)? bool(config["genericDCD"]["FV"]) : true;
        Collision->addGenericDCD(alarm, ms, VF, EE, FV);

        auto gdcd = dynamic_cast<RealSim::collisiondetection::GenericDCD * >(Collision->collisionSet().back().get());
        if(gdcd != nullptr)
        {
            if(config["genericDCD"]["detect_all"] == true)
            {
                std::cout<<GREEN<<"add generic DCD for all object pairs"<<RESET<<std::endl;
                // add all pairs of dynamic-dynamic objects
                for(unsigned int i=0; i<Meshes.size(); i++)
                {
                    for(unsigned int j=i+1; j<Meshes.size(); j++)
                    {
                        gdcd->addObjectPair(Meshes[i].get(), Meshes[j].get(), muVF, muEE, muFV);
                    }
                }

                // add all pairs of static-dynamic objects
//                for(unsigned int i=0; i<Obstacles.size(); i++)
//                {
//                    for(unsigned int j=0; j<Meshes.size(); j++)
//                    {
//                        gdcd->addObjectPair(Obstacles[i].get(), Meshes[j].get(), muVF, muEE, muFV);
//                    }
//                }
            }
            else
            {
                if(config["genericDCD"]["objA"] != nullptr && config["genericDCD"]["objB"] != nullptr)
                {
                    unsigned int objA = config["genericDCD"]["objA"];
                    unsigned int objB = config["genericDCD"]["objB"];

                    if(objA < Meshes.size() && objB < Meshes.size())
                    {
                        std::cout<<GREEN<<"add generic DCD for object pair "<<objA<<" and "<<objB<<std::endl;
                        gdcd->addObjectPair(Meshes[objA].get(), Meshes[objB].get(), muVF, muEE, muFV);
                    }
                }
            }
        }
    }

    if(config["genericCCD"] != nullptr)
    {
        real ms = (config["genericCCD"]["miniseparation"] != nullptr)? real(config["genericCCD"]["miniseparation"]) : 0.0;
        real alarm = (config["genericCCD"]["alarm"] != nullptr)? real(config["genericCCD"]["alarm"]) : ms;
        real muVF = (config["genericCCD"]["muVF"] != nullptr)? real(config["genericCCD"]["muVF"]) : 0.0;
        real muEE = (config["genericCCD"]["muEE"] != nullptr)? real(config["genericCCD"]["muEE"]) : 0.0;
        real muFV = (config["genericCCD"]["muFV"] != nullptr)? real(config["genericCCD"]["muFV"]) : 0.0;
        bool VF = (config["genericCCD"]["VF"] != nullptr)? bool(config["genericCCD"]["VF"]) : true;
        bool EE = (config["genericCCD"]["EE"] != nullptr)? bool(config["genericCCD"]["EE"]) : true;
        bool FV = (config["genericCCD"]["VF"] != nullptr)? bool(config["genericCCD"]["FV"]) : true;
        Collision->addGenericCCD(alarm, ms, VF, EE, FV);

        auto gccd = dynamic_cast<RealSim::collisiondetection::GenericCCD * >(Collision->collisionSet().back().get());
        if(gccd != nullptr)
        {
            if(config["genericCCD"]["detect_all"] == true)
            {
                std::cout<<GREEN<<"add generic CCD for all object pairs"<<RESET<<std::endl;
                // add all pairs of dynamic-dynamic objects
                for(unsigned int i=0; i<Meshes.size(); i++)
                {
                    for(unsigned int j=i+1; j<Meshes.size(); j++)
                    {
                        gccd->addObjectPair(Meshes[i].get(), Meshes[j].get(), muVF, muEE, muFV);
                    }
                }

                // add all pairs of static-dynamic objects
                for(unsigned int i=0; i<Obstacles.size(); i++)
                {
                    for(unsigned int j=0; j<Meshes.size(); j++)
                    {
                        gccd->addObjectPair(Obstacles[i].get(), Meshes[j].get(), muVF, muEE, muFV);
                    }
                }
            }
            else
            {
                if(config["genericCCD"]["objA"] != nullptr && config["genericCCD"]["objB"] != nullptr)
                {
                    unsigned int objA = config["genericCCD"]["objA"];
                    unsigned int objB = config["genericCCD"]["objB"];

                    if(objA < Meshes.size() && objB < Meshes.size())
                    {
                        std::cout<<GREEN<<"add generic CCD for all object "<<objA<<" and "<<objB<<std::endl;
                        gccd->addObjectPair(Meshes[objA].get(), Meshes[objB].get(), muVF, muEE, muFV);
                    }
                }
            }
        }
    }

    if(config["planecollisions"] != nullptr)
    {
        int count = 0;
        for(auto data: config["planecollisions"])
        {
            Vec3R base = Eigen::Map<Vec3R>((std::vector<real>(data["base"])).data(), 3);
            Vec3R normal = Eigen::Map<Vec3R>((std::vector<real>(data["normal"])).data(), 3);
            real mu = (data["mu"] != nullptr)? real(data["mu"]) : 0.0;
            Collision->addPlaneCollision(base, normal, mu);
            auto plane = dynamic_cast<RealSim::collisiondetection::PlaneCollision * >(Collision->collisionSet().back().get());
            if(plane != nullptr)
            {
                if(data["obj"] != nullptr)
                {
                    unsigned int obj = data["obj"];
                    if(obj < Meshes.size()) plane->addCollisionObject(Meshes[obj].get());
                }
                else{
                    std::cout<<GREEN<<"add plane collision for all objects"<<RESET<<std::endl;
                    for (auto & mesh : Meshes)
                    {
                        plane->addCollisionObject(mesh.get());
                    }
                }
            }

            // add visualization
            real scale = (data["visualscale"] != nullptr)? real(data["visualscale"]) : 1.0;
            if(count ==0) LoadPlaneMesh(AnalyticalObstacles, base, getRotation(normal, Vec3R(0,1,0)), Vec3R(scale,scale,scale));
            count++;
        }
    }

    if(config["cylindercollisions"] != nullptr)
    {
        for(auto data: config["cylindercollisions"])
        {
            Vec3R base = Eigen::Map<Vec3R>((std::vector<real>(data["base"])).data(), 3);
            Vec3R axis = Eigen::Map<Vec3R>((std::vector<real>(data["axis"])).data(), 3);
            auto radius = real(data["radius"]);
            real mu = (data["mu"] != nullptr)? real(data["mu"]) : 0.0;
            real avel = (data["rollingvel"] != nullptr)? real(data["rollingvel"]) : 0.0;
            Collision->addCylinderCollision(base, axis, radius, avel, mu);
            auto cylinder = dynamic_cast<RealSim::collisiondetection::CylinderCollision * >(Collision->collisionSet().back().get());
            if(cylinder != nullptr)
            {
                if(data["obj"] != nullptr)
                {
                    unsigned int obj = data["obj"];
                    if(obj < Meshes.size()) cylinder->addCollisionObject(Meshes[obj].get());
                }
                else {
                    std::cout<<GREEN<<"add cylinder collision for all objects"<<RESET<<std::endl;
                    for (auto & mesh : Meshes)
                    {
                        cylinder->addCollisionObject(mesh.get());
                    }
                }

            }
            // add visualization
            real visuallength = (data["visuallength"] != nullptr)? real(data["visuallength"]) : 1.0;
            LoadCylinderMesh(AnalyticalObstacles, base, getRotation(axis, Vec3R(1,0,0)), Vec3R(visuallength, radius*0.98, radius*0.98));
        }
    }

    if(config["spherecollisions"] != nullptr)
    {
        for(auto data: config["spherecollisions"])
        {
            Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(data["center"])).data(), 3);
            auto radius = real(data["radius"]);
            real mu = (data["mu"] != nullptr)? real(data["mu"]) : 0.0;

            if(data["direction"] == nullptr) {
                Collision->addSphereCollision(center, radius, mu);

                auto sphere = dynamic_cast<RealSim::collisiondetection::SphereCollision * >(Collision->collisionSet().back().get());
                if(sphere != nullptr)
                {
                    if(data["obj"] != nullptr)
                    {
                        unsigned int obj = data["obj"];
                        if(obj < Meshes.size()) sphere->addCollisionObject(Meshes[obj].get());
                    }
                    else
                    {
                        std::cout<<GREEN<<"add sphere collision for all objects"<<RESET<<std::endl;
                        for (auto & mesh : Meshes)
                        {
                            sphere->addCollisionObject(mesh.get());
                        }
                    }
                }

                // add visualization
                real visualradius = (data["visualradius"] != nullptr)? real(data["visualradius"]) : radius;
                LoadSphereMesh(AnalyticalObstacles, center, visualradius);
            }
            else {
                std::vector<Vec3R> direction;
                for(unsigned int k=0; k<data["direction"].size(); k++) {
                    direction.emplace_back(Eigen::Map<Vec3R>(std::vector<real>(data["direction"][k]).data(), 3));
                }
                std::vector<real> vel = std::vector<real>(data["vel"]);
                std::vector<real> max = std::vector<real>(data["max"]);

                Collision->addMovingSphereCollision(center, radius, mu, direction, vel, max);

                auto sphere = dynamic_cast<RealSim::collisiondetection::MovingSphereCollision * >(Collision->collisionSet().back().get());
                if(sphere != nullptr)
                {
                    if(data["obj"] != nullptr)
                    {
                        unsigned int obj = data["obj"];
                        if(obj < Meshes.size()) sphere->addCollisionObject(Meshes[obj].get());
                    }
                    else
                    {
                        std::cout << GREEN << "add sphere collision for all objects" << RESET << std::endl;
                        for (auto &mesh: Meshes) {
                            sphere->addCollisionObject(mesh.get());
                        }
                    }
                }
            }
        }
    }

    if(config["toruscollisions"] != nullptr)
    {
        for(auto data: config["toruscollisions"])
        {
            Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(data["center"])).data(), 3);
            Vec3R axis = Eigen::Map<Vec3R>((std::vector<real>(data["axis"])).data(), 3);
            auto radius_outer = real(data["radius_outer"]);
            auto radius_inner = real(data["radius_inner"]);
            real mu = (data["mu"] != nullptr)? real(data["mu"]) : 0.0;

            Collision->addTorusCollision(center, axis, radius_outer, radius_inner, mu);

            auto torus = dynamic_cast<RealSim::collisiondetection::TorusCollision * >(Collision->collisionSet().back().get());
            if(torus != nullptr)
            {
                if(data["obj"] != nullptr)
                {
                    unsigned int obj = data["obj"];
                    if(obj < Meshes.size()) torus->addCollisionObject(Meshes[obj].get());
                }
                else
                {
                    std::cout << GREEN << "add torus collision for all objects" << RESET << std::endl;
                    for (auto &mesh: Meshes) {
                        torus->addCollisionObject(mesh.get());
                    }
                }
            }

            // add visualization
//             LoadTorusMesh(AnalyticalObstacles, center);
        }
    }

    if(config["boxcollisions"] != nullptr)
    {
        for(auto data: config["boxcollisions"])
        {
            Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(data["center"])).data(), 3);
            Vec3R rotation = Eigen::Map<Vec3R>((std::vector<real>(data["rotation"])).data(), 3);
            Vec3R scale = Eigen::Map<Vec3R>((std::vector<real>(data["scale"])).data(), 3);
            real mu = (data["mu"] != nullptr)? real(data["mu"]) : 0.0;

            if(data["direction"] == nullptr) {
                Collision->addBoxCollision(center, rotation, scale, mu);
                auto box = dynamic_cast<RealSim::collisiondetection::BoxCollision * >(Collision->collisionSet().back().get());
                if(box != nullptr)
                {
                    if(data["obj"] != nullptr)
                    {
                        unsigned int obj = data["obj"];
                        if(obj < Meshes.size()) box->addCollisionObject(Meshes[obj].get());
                    }
                    else
                    {
                        std::cout << GREEN << "add square box collision for all objects" << RESET << std::endl;
                        for (auto &mesh: Meshes) {
                            box->addCollisionObject(mesh.get());
                        }
                    }
                }

                // add visualization
                LoadBoxMesh(AnalyticalObstacles, center, rotation, scale);
            }
            else {
                std::vector<Vec3R> direction;
                for(unsigned int k=0; k<data["direction"].size(); k++) {
                    direction.emplace_back(Eigen::Map<Vec3R>(std::vector<real>(data["direction"][k]).data(), 3));
                }
                std::vector<real> vel = std::vector<real>(data["vel"]);
                std::vector<real> max = std::vector<real>(data["max"]);

                Collision->addMovingBoxCollision(center, rotation, scale, mu, direction, vel, max);

                auto box = dynamic_cast<RealSim::collisiondetection::MovingBoxCollision * >(Collision->collisionSet().back().get());
                if(box != nullptr)
                {
                    if(data["obj"] != nullptr)
                    {
                        unsigned int obj = data["obj"];
                        if(obj < Meshes.size()) box->addCollisionObject(Meshes[obj].get());
                    }
                    else
                    {
                        std::cout << GREEN << "add square box collision for all objects" << RESET << std::endl;
                        for (auto &mesh: Meshes) {
                            box->addCollisionObject(mesh.get());
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int InitController(const std::shared_ptr<RealSim::lagrange::ControlHandler>& Controller, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Meshes, std::vector<json> &objectsconfig)
{
    for(int i=0; i<objectsconfig.size(); i++)
    {
        json config = objectsconfig[i];
        if(config["cable"] != nullptr)
        {
            auto tetra_mesh = dynamic_cast<RealSim::geometry::STetrahedraMesh *>(Meshes[i].get());
            if(tetra_mesh != nullptr)
            {
                real invstiff = (config["cableInvStiff"] != nullptr)? real(config["cableInvStiff"]) : 0.0;

                MatXR pos_cable(config["cable"].size(), 3);
                for(unsigned int k=0; k<config["cable"].size(); k++)
                {
                    pos_cable.row(k) = Eigen::Map<Vec3R>(std::vector<real>(config["cable"][k]).data(), 3);
                }

                Vec3R translation = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["trans"])).data(), 3);
                Vec3R rotation = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["rotation"])).data(), 3);
                Vec3R scale = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["scale"])).data(), 3);
                Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(config["transformation"]["center"])).data(), 3);

                SimulationData Data;
                Data.addVertices(pos_cable, translation, rotation, scale, center);

                Controller->addCableController(Data.positions, invstiff);
                auto cable = dynamic_cast<RealSim::lagrange::CableController * >(Controller->controllerSet().back().get());
                if(cable != nullptr)
                {
                    std::cout<<GREEN<<"add cable for object "<<i<<RESET<<std::endl;
                    cable->mappingToTetrahedraMesh(*tetra_mesh->getPositions(), *tetra_mesh->getTetrahedra());
                }
            }
        }
    }

    return 0;
}

int InitIntegrator(const std::shared_ptr<RealSim::integrator::projectivedynamics::LocalGlobalSolver>& solver, SimulationData& Data, std::vector<std::shared_ptr<RealSim::geometry::SBaseGeometry>> & Meshes, json & simconfig, std::vector<json> &objectsconfig)
{
    real dt = real(simconfig["timestep"]);
    solver->setDt(dt);

    solver->setVertexDimension(Data.positions.rows());
    solver->setMass();

    Vec3R gravity = Eigen::Map<Vec3R>((std::vector<real>(simconfig["gravity"])).data(), 3);
    solver->setGravity(gravity);

    for(int i=0; i<Data.num_meshes(); i++)
    {
        json config = objectsconfig[i];

        real mass = config["mechanical_props"]["obj_mass"];
        real YoungModulus = config["mechanical_props"]["young"];
        real PoissonRatio = config["mechanical_props"]["poisson"];

        std::string constitutive = config["mechanical_props"]["constitutive"];

        ////add mass
        solver->addObjectMass(mass, Data.getVertices()[i]);

        real bending = (config["mechanical_props"]["bending"] != nullptr)? real(config["mechanical_props"]["bending"]) : 0.0;
        std::string bendingEnergy = (config["mechanical_props"]["bending_type"] != nullptr)? std::string(config["mechanical_props"]["bending_type"]) : "BEND_ISOMETRIC";

        ////add elastic energy
        solver->addElasticEnergy(RealSim::integrator::projectivedynamics::energy::string2Model(constitutive), Meshes[i].get(), YoungModulus, PoissonRatio, bending, RealSim::integrator::projectivedynamics::energy::string2Model(bendingEnergy));

        if(config["node_pin"] != nullptr)
        {
            auto node = std::vector<int>(config["node_pin"]);
            VecXI pins = Eigen::Map<VecXI>(node.data(), node.size());
            solver->addPinEnergy(1e10, pins, Data.positions);
        }

        if(config["pin"] != nullptr)
        {
            for(auto pin_config: config["pin"])
            {
                VecXI pins;

                if(pin_config["box"] != nullptr)
                {
                    auto box = std::vector<real>(pin_config["box"]);
                    std::shared_ptr<RealSim::tools::AxisAlignedBoundingBox> AABB = std::make_shared<RealSim::tools::AxisAlignedBoundingBox>(box, Meshes[i].get());
                    pins = AABB->verticesInside();
                }
                else if(pin_config["node"] != nullptr)
                {
                    auto node = std::vector<int>(config["node"]);
                    pins = Eigen::Map<VecXI>(node.data(), node.size());
                }
                else continue;

                if(pin_config["dynamic"] != nullptr)
                {
                    auto PinEnergy = std::make_shared<RealSim::integrator::projectivedynamics::energy::DynamicPinEnergy>(1e10, pins, Data.positions);

                    std::vector<std::shared_ptr<RealSim::tools::action::Action>> actions;
                    for(auto action: pin_config["dynamic"])
                    {
                        auto type = std::string(action["action"]);
                        if(type == "PULLING")
                        {
                            Vec3R direction = Eigen::Map<Vec3R>((std::vector<real>(action["direction"])).data(), 3);
                            auto vel = real(action["vel"]);
                            auto maxlength = real(action["maxlength"]);

                            actions.push_back(std::make_shared<RealSim::tools::action::PullingAction>(vel*dt, maxlength, direction));
                        }
                        else if(type == "ROLLING")
                        {
                            Vec3R center = Eigen::Map<Vec3R>((std::vector<real>(action["center"])).data(), 3);
                            Vec3R axis = Eigen::Map<Vec3R>((std::vector<real>(action["axis"])).data(), 3);
                            auto avel = real(action["avel"]);
                            auto maxrotation = real(action["maxrotation"]);

                            actions.push_back(std::make_shared<RealSim::tools::action::RollingAction>(avel*dt, maxrotation, center, axis));
                        }
                    }

                    if(!actions.empty()) solver->addDynamicPinEnergy(1e10, pins, Data.positions, actions);
                    else solver->addPinEnergy(1e10, pins, Data.positions);
                }
                else
                {
                    solver->addPinEnergy(1e10, pins, Data.positions);
                }
            }
        }

        if(config["axis_pin"] != nullptr)
        {
            Vec3R base = Eigen::Map<Vec3R>((std::vector<real>(config["axis_pin"]["base"])).data(), 3);
            Vec3R axis = Eigen::Map<Vec3R>((std::vector<real>(config["axis_pin"]["axis"])).data(), 3);
            auto radius = real(config["axis_pin"]["radius"]);
            auto slide = bool(config["axis_pin"]["slide"]);
            solver->addAxisPinEnergy(1e10, Data.positions, base, axis, radius, slide);
        }
    }

    ////add linear solver for global step
    std::string linearsolver = simconfig["linearsolver"]["type"];
    solver->addLinearSolver(RealSim::linearsolver::sparse::string2Type(linearsolver));

    if(simconfig["linearsolverref"] != nullptr)
    {
        std::string linearsolverref = simconfig["linearsolverref"]["type"];
        solver->addLinearSolver(RealSim::linearsolver::sparse::string2Type(linearsolverref), true);
    }
//    solver->setCGLinearSolver(simconfig["linearsolver"]["iterations"], simconfig["linearsolver"]["tolerance"], simconfig["linearsolver"]["warmstart"]);

    ////add constraint solver
    auto constraintsolver = simconfig["constraintsolver"];
    if(constraintsolver != nullptr)
    {
        std::string type = constraintsolver["type"];
        std::cout<<GREEN<<"add constraint solver "<<type<<RESET<<std::endl;
        std::string function = constraintsolver["function"];
        if(RealSim::lagrange::constraintsolver::string2CSolverType(type) == RealSim::lagrange::constraintsolver::TypeConstraintSolver::NonSmoothNewton)
            std::cout<<GREEN<<"non-smooth function "<<function<<RESET<<std::endl;
        solver->addConstraintSolver(RealSim::lagrange::constraintsolver::string2CSolverType(type));

        std::string constraintlinearsolver = simconfig["constraintsolver"]["linearsolver"];
        solver->setConstraintSolver(RealSim::linearsolver::dense::string2Type(constraintlinearsolver), constraintsolver["iterations"], constraintsolver["tolerance"], constraintsolver["maxforce"], RealSim::lagrange::constraintsolver::string2NNFuncType(function));
    }

    int timer = (simconfig["timer"] != nullptr)? int(simconfig["timer"]) : 1;
    solver->setTimerPrint(timer);

    solver->init(Data.positions);

    return 0;
}

