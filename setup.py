from distutils.core import setup

setup(
    name="Fit2D",
    version="0.1.0",
    author="Anna Kwa",
    author_email="anna.s.kwa@gmail.com",
    packages=["fit2d"],
    license="LICENSE.txt",
    description="Model galaxy 1st moment maps with MCMC packages.",
    install_requires=["astropy", "emcee==2.2.1", "missingpy", "numpy", "scipy",],
)
