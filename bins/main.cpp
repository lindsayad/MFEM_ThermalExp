#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "DomainLFH1DivIntegrator.hpp"
#include "MixedWeakDivergenceIntegrator.hpp"




int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);                                                                                           
   int num_procs = mfem::Mpi::WorldSize();                                                                                
   int myid = mfem::Mpi::WorldRank();                                                                                     
   mfem::Hypre::Init();    

   const char *mesh_file = "../simple_cube.g";
   int order = 1;

   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   int dim = mesh->Dimension();
   mfem::Coefficient *hot = new mfem::ConstantCoefficient(500);
   mfem::Coefficient *cold = new mfem::ConstantCoefficient(300);

   mfem::Coefficient *thermalConductivityCoef = new mfem::ConstantCoefficient(300); //
   mfem::Coefficient *lameCoef = new mfem::ConstantCoefficient(10);  // Lame's first parameter
   mfem::Coefficient *shearModulusCoef = new mfem::ConstantCoefficient(10);  // Shear modulus 
   mfem::Coefficient *thermalExpansionCoef = new mfem::ConstantCoefficient(0.00002);  // Thermal expansion coefficient
   mfem::Coefficient *stressFreeTempCoef = new mfem::ConstantCoefficient(300.0);  // Stress free temperature
   mfem::Coefficient *zero = new mfem::ConstantCoefficient(0);  // Zero
   mfem::Coefficient *materialTerm = new mfem::SumCoefficient(*lameCoef, *shearModulusCoef, 3, 2); // Material term in lame params (3λ + 2μ)
   mfem::Coefficient *thexpStressFreeTemp = new mfem::ProductCoefficient(*thermalExpansionCoef, *stressFreeTempCoef); // Thermal Expansion Coefficient * Lame Params
   mfem::Coefficient *bilinearFormCoefPositive = new mfem::ProductCoefficient(*thermalExpansionCoef, *materialTerm);  
   mfem::Coefficient *bilinearFormCoefNegative = new mfem::ProductCoefficient(-1.0, *bilinearFormCoefPositive);  
   mfem::Coefficient *thexpLinearCoef = new mfem::ProductCoefficient(*thexpStressFreeTemp, *materialTerm);  

   mfem::VectorArrayCoefficient zero_vec(3);

   // Set up FEC
   mfem::FiniteElementCollection *fec;
   fec = new mfem::H1_FECollection(order, dim);
   

   mfem::ParFiniteElementSpace* tempFESpace = new mfem::ParFiniteElementSpace(pmesh, fec, 1);   
   mfem::ParFiniteElementSpace* dispFESpace = new mfem::ParFiniteElementSpace(pmesh, fec, 3, mfem::Ordering::byVDIM);   


   mfem::ParGridFunction t_(tempFESpace);
   mfem::ParGridFunction u_(dispFESpace);
   mfem::Array<int> ess_temp_bdr_tdofs, ess_disp_bdr_tdofs;  

   mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   t_.ProjectBdrCoefficient(*cold, ess_bdr);
   u_.ProjectBdrCoefficient(zero_vec, ess_bdr);
   u_.FESpace()->GetEssentialTrueDofs(ess_bdr, ess_disp_bdr_tdofs);
   // u_.ProjectBdrCoefficient(*zero, ess_bdr);

   ess_bdr = 0;
   ess_bdr[1] = 1;
   t_.ProjectBdrCoefficient(*hot, ess_bdr);
   

   ess_bdr = 1; //Now both boundary attributes are set to 1, so we get correct true dofs for temp
   t_.FESpace()->GetEssentialTrueDofs(ess_bdr, ess_temp_bdr_tdofs);


   // Set up bilinear and mixed bilinear forms
   mfem::ParBilinearForm *temp_bf = new mfem::ParBilinearForm(tempFESpace);
   mfem::ParBilinearForm *disp_bf = new mfem::ParBilinearForm(dispFESpace);
   mfem::ParMixedBilinearForm *mixed_bf = new mfem::ParMixedBilinearForm(tempFESpace, dispFESpace);

   // Set up linear forms for 
   mfem::ParLinearForm *temp_lf = new mfem::ParLinearForm(t_.ParFESpace());
   mfem::ParLinearForm *disp_lf = new mfem::ParLinearForm(u_.ParFESpace());

   mixed_bf->AddDomainIntegrator(new mfem::MixedWeakDivergenceIntegrator(*bilinearFormCoefNegative));
   temp_bf->AddDomainIntegrator(new mfem::DiffusionIntegrator(*thermalConductivityCoef));
   disp_bf->AddDomainIntegrator(new mfem::ElasticityIntegrator(*lameCoef, *shearModulusCoef));

   disp_lf->AddDomainIntegrator(new mfem::DomainLFH1DivIntegrator(*thexpLinearCoef));
   temp_lf->AddDomainIntegrator(new mfem::DomainLFIntegrator(*zero));

   mixed_bf->Assemble();
   temp_bf->Assemble();
   disp_bf->Assemble();
   
   mixed_bf->Finalize();
   temp_bf->Finalize();
   disp_bf->Finalize();

   disp_lf->Assemble();
   temp_lf->Assemble();
   
   mfem::Array<int> offsets({0, t_.ParFESpace()->TrueVSize(), 
      t_.ParFESpace()->TrueVSize() + u_.ParFESpace()->TrueVSize()});

   mfem::BlockVector trueX(offsets);
   mfem::BlockVector trueRHS(offsets);

   mfem::Array2D<mfem::HypreParMatrix *> OpBlocks;
   OpBlocks.DeleteAll();
   OpBlocks.SetSize(2,2);
   OpBlocks(0, 0) = new mfem::HypreParMatrix;
   OpBlocks(1, 1) = new mfem::HypreParMatrix;
   OpBlocks(1, 0) = new mfem::HypreParMatrix;
   OpBlocks(0, 1) = nullptr;

   temp_bf->FormLinearSystem(ess_temp_bdr_tdofs, t_, *temp_lf, *OpBlocks(0, 0), trueX.GetBlock(0), trueRHS.GetBlock(0));
   disp_bf->FormLinearSystem(ess_disp_bdr_tdofs, u_, *disp_lf, *OpBlocks(1, 1), trueX.GetBlock(1), trueRHS.GetBlock(1));
   mixed_bf->FormRectangularLinearSystem(ess_temp_bdr_tdofs, ess_disp_bdr_tdofs, t_, *disp_lf, *OpBlocks(1, 0), trueX.GetBlock(0), trueRHS.GetBlock(1));

   mfem::HypreParMatrix *A1 = mfem::HypreParMatrixFromBlocks(OpBlocks);
   mfem::HypreBoomerAMG *amg = new mfem::HypreBoomerAMG(*A1);
   mfem::HyprePCG solver(MPI_COMM_WORLD);
   
   solver.SetOperator(*A1);
   solver.SetTol(1e-15);                                                                                               
   solver.SetMaxIter(500);                                                                                            
   solver.SetPrintLevel(1);                                                                                           
   solver.SetPreconditioner(*amg);                                                                                    

   solver.Mult(trueRHS, trueX);

 


   mfem::ParaViewDataCollection* pd = NULL;
   pd = new mfem::ParaViewDataCollection("ExampleThermalExpansion", pmesh);
   pd->SetPrefixPath("ParaView");
   pd->SetTime(0.0);
   
   pd->RegisterField("disp", &u_);
   pd->RegisterField("temp", &t_);

   temp_bf->RecoverFEMSolution(trueX.GetBlock(0), *temp_lf, t_);        
   disp_bf->RecoverFEMSolution(trueX.GetBlock(1), *disp_lf, u_);        

   pd->SetTime(1.0);
   pd->RegisterField("disp", &u_);
   pd->RegisterField("temp", &t_);
   pd->SetDataFormat(mfem::VTKFormat::BINARY);
   
   
   pd->Save();
   delete pd;
   

   delete(amg);
   delete(A1);
   
   OpBlocks.DeleteAll();


   delete mesh;

   return 0;
}
