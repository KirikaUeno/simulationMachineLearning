package ui;

import data.SimulationParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;

public class StaticItemPanel extends JPanel {
    public JTable staticItemTable;
    public ArrayList<SimulationParameter> staticParameters = new ArrayList<>();

    public StaticItemPanel(){
        setPreferredSize(new Dimension(Config.boardWidth*2/5, Config.boardHeight));
        setFocusable(true);
        setName("staticItemPanel");
        initializeVariables();
    }

    private void initializeVariables(){
        staticItemTable = new JTable();
        DefaultTableModel dtm = new StaticTable(staticParameters);
        staticItemTable.setModel(dtm);
        dtm.addTableModelListener(staticItemTable);
        staticItemTable.setDefaultRenderer(Object.class, new ParameterTableRenderer());


        JScrollPane scrollPane = new JScrollPane(staticItemTable);

        SpringLayout layoutMain = new SpringLayout();
        layoutMain.putConstraint(SpringLayout.NORTH, scrollPane, 0, SpringLayout.NORTH, this);
        layoutMain.putConstraint(SpringLayout.SOUTH, scrollPane, 0, SpringLayout.SOUTH, this);
        layoutMain.putConstraint(SpringLayout.WEST, scrollPane, 0, SpringLayout.WEST, this);
        layoutMain.putConstraint(SpringLayout.EAST, scrollPane, 0, SpringLayout.EAST, this);
        setLayout(layoutMain);

        add(scrollPane);
    }
}
