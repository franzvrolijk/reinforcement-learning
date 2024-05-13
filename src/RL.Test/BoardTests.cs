using ReinforcementLearning;

namespace RL.Test;

public class BoardTests
{
    [Fact]
    public void BoardTest()
    {
        var board = new Board(10, 10, (5, 5), (1, 1));

        var positions = board.GetPositions();
        var (currentX, currentY) = (positions[2], positions[3]);

        Assert.Equal(1, currentX);
        Assert.Equal(1, currentY);

        board.Move(0);

        positions = board.GetPositions();
        (currentX, currentY) = (positions[2], positions[3]);

        Assert.Equal(2, currentX);
        Assert.Equal(1, currentY);

        board.Move(0.25);

        positions = board.GetPositions();
        (currentX, currentY) = (positions[2], positions[3]);

        Assert.Equal(2, currentX);
        Assert.Equal(2, currentY);
    }
}