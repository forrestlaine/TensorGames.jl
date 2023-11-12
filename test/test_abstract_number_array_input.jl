# The package should be able to handle input that is any abstract array of Numbers

@testset "non-float input" begin
    # Matching pennies with integers
    A = [-1 1; 1 -1]
    B = -A
    @test compute_equilibrium([A, B]).x == [[0.5, 0.5], [0.5, 0.5]]
end

@testset "abstract array input" begin
    # Prisoner's dilemma using transpose
    A = [2 0; 3 1]
    B = A'
    @test compute_equilibrium([A, B]).x == [[0, 1], [0, 1]]
end
