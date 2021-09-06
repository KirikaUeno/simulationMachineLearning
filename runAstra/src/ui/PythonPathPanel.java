package ui;

import javax.swing.*;
import java.awt.*;

public class PythonPathPanel extends JPanel {
    public JLabel specifyPath = new JLabel("specify path to your python (it should have torch, sklearn and matplotlib): ");
    public TextField pythonPath = new TextField("python");

    public PythonPathPanel(){
        setPreferredSize(new Dimension(600, 55));
        setBackground(Color.LIGHT_GRAY);
        pythonPath.setPreferredSize(new Dimension(400,20));
        setFocusable(true);
        setName("python path panel");
        initializeVariables();
    }

    public void initializeVariables(){
        add(specifyPath);
        add(pythonPath);
    }
}
