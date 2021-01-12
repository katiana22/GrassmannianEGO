#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>

#include "common.hh"
#include "extra.hh"
#include "shear_sim.hh"

// Temperature scale that is used to non-dimensionalize temperatures
const double TZ=21000;                   // STZ formation energy, eZ/kB [K]
const double kB=1.3806503e-23;           // Boltzmann constant [J/K]
const double kB_metal=8.617330350e-5;    // Boltzmann constant [eV/K]
const double eV_per_J=6.241509e18;       // Converts eV to Joules
const double Ez=TZ*kB*eV_per_J;          // STZ formation Energy [eV]

void syntax_message() {
    fputs("Syntax: ./shear_energy_Adam <run_type> <input_file> <beta> <u0> [additional arguments]\n\n"
          "<run_type> is either:\n"
          "     \"direct\" for direct simulation,or\n"
          "     \"qs\" for quasi-statc simulation\n\n"
          "<input_file> expects the relative path for a text file containing potential energy data\n\n"
          "<beta> is the initial energy scaling parameter [1/eV]\n\n"
          "<u0> is an energy offset value [eV]\n",stderr);
    exit(1);
}

int main(int argc,char **argv) {

    // Check command-line arguments
    if(argc<5) syntax_message();
    bool qs=true;
    if(strcmp(argv[1],"direct") == 0) qs=false;
    else if(strcmp(argv[1],"qs") != 0) syntax_message();
    double beta=atof(argv[3]),u0=atof(argv[4]);

    printf("Beta is set to %g 1/eV\n", beta);
    printf("u0 is set to %g eV\n", u0);

    // Set default values of parameters
    double rho0=6125;           // Density (kg/m^3)
    double s_y=0.85;              // Yield stress (GPa)
    double l_chi=4.01;
    double c0=0.3;                // Plastic work fraction
    double ep=10;
    double chi_inf=2730;   

    //char dfn_default[]="sct_d.out";
    //char qfn_default[]="sct_q.out";
    char er_default[]="pe.Cu.MD.100.txt";

    char* endref=er_default;
    //char* qfn_=qfn_default;
    //char* dfn_=dfn_default;

    // Read additional command-line arguments to override default parameter values
    int i=5;
    while(i<argc) {
        if(read_arg(argv,"chi_inf",i,argc,chi_inf,"steady-state effective temperature"," K")) {}
        else if(read_arg(argv,"chi_len",i,argc,l_chi,"chi diffusion length scale"," Angstroms")) {}
        else if(read_arg(argv,"c0",i,argc,c0,"plastic work fraction","")) {}
        else if(read_arg(argv,"ep",i,argc,ep,"STZ size","")) {}
        else if(read_arg(argv,"s_y",i,argc,s_y,"yield stress","")) {}
  
        //else if(se(argv[i],"outdir")) {
        //    if(++i==argc) {
        //        fputs("Error reading command-line arguments\n", stderr);
        //        return 1;
        //    }
        //    qfn_ = argv[i];
        //}
        else if(se(argv[i],"endref")) {
            if(++i==argc) {
                fputs("Error reading command-line arguments\n",stderr);
                return 1;
            }
            printf("Reading final state from file %s\n",endref=argv[i]);
        } else {
            fprintf(stderr,"Command-line argument '%s' not recognized\n",argv[i]);
        }
        i++;
    }

    // If chi_inf was specified on the command line, then use that value.
    if(chi_inf>0) {
        printf("\nThe upper-limiting effective temperature was set using a user-defined value.\n"
               "The effective temperature is %g K.\n",chi_inf);
        chi_inf/=TZ;
    } else {

        // Open final MD PE reference file and determine maximum PE
        FILE *fp=safe_fopen(endref,"r");
        double PE,maxPE=0;
        while(fscanf(fp,"%lf",&PE)) {
            if(PE>maxPE) maxPE=PE;
        }
        fclose(fp);

        // Compute value of chi_infinity from maximum potential energy
        chi_inf=0.95*beta*(maxPE-u0)*Ez/kB_metal/TZ;
        printf("\nThe maximum PE from the final snapshot is %g eV.\n"
               "For beta=%g and E0=%g the corresponding dimensionless value of"
               "chi_inf is %g. This is %g K.\n",maxPE,beta,u0,chi_inf,chi_inf*TZ);
    }

    // Rescaled elasticity parameters
    const double mu_phys = 20;
    const double mu = mu_phys/s_y;           // Shear modulus (GPa)
    const double nu = 0.35;                     // Poisson's ratio (--)
    const double K = 2*mu*(1+nu)/(3*(1-2*nu));  // Bulk modulus (GPa)

    // STZ model parameters (based on Vitreloy 1 BMG)
    const double tau0=1e-13;                  // Vibration timescale (s)

    // Output filenames
    const char dfn[]="sct_d.out",qfn[]="sct_q.out";
    //const char* dfn=dfn_;
    //const char* qfn=qfn_;

    // Output fields and miscellaneous flags. 1-u,2-v,4-p,8-q,16-s,32-tau,
    // 64-chi,128-tem,256-dev,512-X,1024-Y,2048-(total strain components),
    // 4096-(total strain invariants),8192-(Lagrangian tracer output).
    const unsigned int fflags=4|8|16|32|64|128|256|512|1024|2048|8192;

    // Other parameters. The scale factor applies a scaling to the rate of
    // plasticity and the applied strain. It is used to allow for comparison
    // between the explicit and quasi-static models.
    const double le=1e-10;                   // Length scale (m)
    const double sca=2e4;                   // Scale factor [only used in non-periodic test]
    const double visc=0;                 // Viscous damping
    // const double chi_len=l_chi*1e-10;    // Dpl-mediated diffusion (m)
    const double chi_len=l_chi;             // Dpl-mediated diffusion (m)
    const double adapt_fac=2e-3;            // Adaptivity factor

    // MD simulation-based continuum parameters
    const int x_grid=32;              // Grid points in x-direction
    const int y_grid=32;              // Grid points in y-direction
    const double x_beg=0.0;           // x-origin (m)
    const double x_end=4e-8;          // x-terminus (m)
    const double y_beg=0.0;           // y-origin (m)
    const double y_end=4e-8;          // y-terminus (m)
    const double gam_max=0.5;         // MD max strain

    // Compute conversion factor from simulation time to physical time. This
    // enters into the plasticity model. This value is determined automatically
    // from the length scale, density, and shear modulus in order to make shear
    // waves move at speed 1 in the simulation units.
    const double t_scale=le/sqrt(mu*s_y/rho0);

    // Set parameters for either a periodic or non-periodic test
    const bool y_prd=true;
    double u_bdry,lamb,lamb_phys,tf;
    if(y_prd) {

        // Periodic test
        u_bdry=0.;
        lamb_phys=1e8;                  // Strain rate (1/s)
        lamb=1e8*t_scale;               // Strain rate in simulation units
        tf=gam_max/lamb;                // Final time
    } else {

        // Non-periodic test
        u_bdry=1e-7*sca;
        lamb=0;tf=3e6/sca;
    }

    // Make the output directory if it doesn't already exist
    mkdir(qs?qfn:dfn,S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH);

    // Initialize an STZ plasticity model
    stz_dynamics_adam stz(TZ,chi_inf,tau0,ep,c0);    

    // Initialize the simulation
    shear_sim sim(x_grid,y_grid,x_beg/le,x_end/le,y_beg/le,y_end/le,mu,K,
                  visc,chi_len,t_scale,adapt_fac,u_bdry,lamb,&stz,y_prd,fflags,qs? qfn : dfn);
    sim.init_fields(0,580,220);
    //sim.initialize_tracers(32,32);    

    // Open the input file and read in the grid dimensions and scaling
    read_chi_from_file(argv[2],sim,u0,beta*Ez/kB_metal,TZ);
    //sim.initialize_random(5,580,20);

    // Carry out the simulation using the selected simulation method
    int n_frames=100,steps=60;
    printf("Simulation time unit   : %g s\n",t_scale);
    printf("Final time             : %g (%g s) \n",tf,tf*t_scale);
    printf("Quasi-static step size : %g \n",tf/(n_frames*steps));
    qs?sim.solve_quasistatic(tf,n_frames,steps):sim.solve(tf,160);
}
