package ui;

import data.InputParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;

public class InputParametersPanel extends JPanel{
    private final MainPanel mainPanel;
    public JTable table;
    public ArrayList<InputParameter> parameters = new ArrayList<>();

    public InputParametersPanel(MainPanel mainPanel){
        setPreferredSize(new Dimension(Config.boardWidth/2, Config.boardHeight));
        setFocusable(true);
        setName("input panel");
        this.mainPanel = mainPanel;
        initializeVariables();
    }

    private void initializeVariables(){
        table = new JTable();
        DefaultTableModel dtm = new InputParametersTable(parameters);
        table.setModel(dtm);
        dtm.addTableModelListener(table);
        table.setDefaultRenderer(Object.class, new ParameterTableRenderer());


        JScrollPane scrollPane = new JScrollPane(table);

        SpringLayout layoutMain = new SpringLayout();
        layoutMain.putConstraint(SpringLayout.NORTH, scrollPane, 0, SpringLayout.NORTH, this);
        layoutMain.putConstraint(SpringLayout.SOUTH, scrollPane, 0, SpringLayout.SOUTH, this);
        layoutMain.putConstraint(SpringLayout.WEST, scrollPane, 0, SpringLayout.WEST, this);
        layoutMain.putConstraint(SpringLayout.EAST, scrollPane, 0, SpringLayout.EAST, this);
        setLayout(layoutMain);

        add(scrollPane);
    }
}
