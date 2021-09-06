package ui;

import data.InputParameter;
import data.OutputParameter;
import data.SimulationParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Random;

public class MainPanel extends JPanel {
    public final StaticItemPanel staticItemPanel = new StaticItemPanel();
    public final DynamicItemPanel dynamicItemPanel = new DynamicItemPanel();
    public final PythonPathPanel pythonPathPanel = new PythonPathPanel();
    private final SpringLayout layout = new SpringLayout();
    private final TextField pathField = new TextField("run.in");
    private final TextField outMidName = new TextField("0528");
    private final Button loadPath = new Button("Load in file");
    private final Button sendRandomJob = new Button("Send random-sampled job");
    private final Button sendSteppedJob = new Button("Send stepped job");
    private final JLabel samplesNumber = new JLabel("samples number:");
    private final TextField samplesNumberField = new TextField("");
    private final JButton switchToML = new JButton("Make a prediction");
    private final JButton switchToSimulation = new JButton("Make simulation scan");
    private final Button moveLeft = new Button("<-");
    private final Button moveRight = new Button("->");
    public ArrayList<SimulationParameter> allParameters = new ArrayList<>();

    public final InputParametersPanel inputPanel = new InputParametersPanel();
    public final OutputParameterPanel outputPanel = new OutputParameterPanel();
    public final Button trainAndPredict = new Button("train and predict");
    public final Button predict = new Button("predict");
    public final JLabel chooseModel = new JLabel("choose model");
    public final Choice model = new Choice();
    public final TextField pathMLField = new TextField("/");
    public final JLabel enterOutMidName = new JLabel("z-coordinate");
    public final Button loadResults = new Button("calculate dataset from outputs");
    public final Button loadInputs = new Button("just load inputs");
    public final JLabel getAllParamsFrom = new JLabel("get all other plot params from simulation :");
    public final TextField paramsIndex = new TextField("0");
    public final Button draw = new Button("draw train results");
    public final Button showPythonPath = new Button("PY");

    private int fileNumber = 0;
    //"/afs/ifh.de/user/k/kladov/volume/pythonLocal/bin/python3.9"
    private boolean pythonPathShowingState = false;

    public MainPanel() {
        setPreferredSize(new Dimension(Config.boardWidth, Config.boardHeight));

        setFocusable(true);
        setName("mainPanel");
        initializeVariables();
        repaint();
    }

    private void initializeVariables(){
        pathField.setPreferredSize(new Dimension(200,25));
        samplesNumberField.setPreferredSize(new Dimension(40,20));
        loadPath.addActionListener(e -> loadList(pathField.getText()));
        sendSteppedJob.addActionListener(e -> {sendSteppedJob(0,new ArrayList<>());writeInputParameters();fileNumber=0;});
        sendRandomJob.addActionListener(e -> sendRandomJob());
        moveLeft.addActionListener(e->moveSelectedLeft());
        moveRight.addActionListener(e->moveSelectedRight());
        switchToML.addActionListener(e->switchToMLAction());
        switchToSimulation.addActionListener(e->switchToSimulationAction());
        loadResults.addActionListener(e->loadPath());
        loadInputs.addActionListener(e->loadInputs());
        trainAndPredict.addActionListener(e->trainAndPredict());
        predict.addActionListener(e->predict());
        draw.addActionListener(e->draw());
        showPythonPath.addActionListener(e->{pythonPathPanel.setVisible(!pythonPathShowingState);
            pythonPathShowingState = !pythonPathShowingState; inputPanel.setVisible(!pythonPathShowingState);
            outputPanel.setVisible(!pythonPathShowingState);});

        model.add("NN");
        model.add("Tree");
        model.add("Tree_Boost");
        model.add("NN+Boost");
        model.select(1);

        layout.putConstraint(SpringLayout.WEST, pathField, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.NORTH, pathField, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.WEST, loadPath, 15, SpringLayout.EAST, pathField);
        layout.putConstraint(SpringLayout.NORTH, loadPath, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.EAST, sendRandomJob, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.SOUTH, sendRandomJob, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.EAST, samplesNumber, -5, SpringLayout.WEST, samplesNumberField);
        layout.putConstraint(SpringLayout.SOUTH, samplesNumber, -5, SpringLayout.NORTH, sendRandomJob);
        layout.putConstraint(SpringLayout.EAST, samplesNumberField, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.SOUTH, samplesNumberField, -5, SpringLayout.NORTH, sendRandomJob);
        layout.putConstraint(SpringLayout.EAST, sendSteppedJob, -5, SpringLayout.WEST, sendRandomJob);
        layout.putConstraint(SpringLayout.SOUTH, sendSteppedJob, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, moveLeft, 5, SpringLayout.WEST, staticItemPanel);
        layout.putConstraint(SpringLayout.NORTH, moveLeft, 5, SpringLayout.SOUTH, staticItemPanel);
        layout.putConstraint(SpringLayout.EAST, moveRight, -5, SpringLayout.EAST, dynamicItemPanel);
        layout.putConstraint(SpringLayout.NORTH, moveRight, 5, SpringLayout.SOUTH, dynamicItemPanel);
        layout.putConstraint(SpringLayout.EAST, switchToML, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, switchToML, 5, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.EAST, switchToSimulation, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, switchToSimulation, 5, SpringLayout.NORTH, this);

        layout.putConstraint(SpringLayout.EAST, staticItemPanel, 0, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, staticItemPanel, 40, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, staticItemPanel, -80, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.EAST, dynamicItemPanel, 0, SpringLayout.WEST, staticItemPanel);
        layout.putConstraint(SpringLayout.NORTH, dynamicItemPanel, 40, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, dynamicItemPanel, -80, SpringLayout.SOUTH, staticItemPanel);
        layout.putConstraint(SpringLayout.WEST, dynamicItemPanel, 0, SpringLayout.WEST, this);

        layout.putConstraint(SpringLayout.EAST, outputPanel, 0, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, outputPanel, 65, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, outputPanel, -80, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.EAST, inputPanel, 0, SpringLayout.WEST, outputPanel);
        layout.putConstraint(SpringLayout.NORTH, inputPanel, 65, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, inputPanel, -80, SpringLayout.SOUTH, outputPanel);
        layout.putConstraint(SpringLayout.WEST, inputPanel, 0, SpringLayout.WEST, this);

        layout.putConstraint(SpringLayout.WEST, enterOutMidName, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.NORTH, enterOutMidName, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.WEST, outMidName, 5, SpringLayout.EAST, enterOutMidName);
        layout.putConstraint(SpringLayout.NORTH, outMidName, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.WEST, pathMLField, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.NORTH, pathMLField, 10, SpringLayout.SOUTH, enterOutMidName);
        layout.putConstraint(SpringLayout.WEST, loadResults, 5, SpringLayout.EAST, pathMLField);
        layout.putConstraint(SpringLayout.NORTH, loadResults, 10, SpringLayout.SOUTH, enterOutMidName);
        layout.putConstraint(SpringLayout.WEST, loadInputs, 5, SpringLayout.EAST, loadResults);
        layout.putConstraint(SpringLayout.NORTH, loadInputs, 10, SpringLayout.SOUTH, enterOutMidName);
        layout.putConstraint(SpringLayout.EAST, showPythonPath, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, showPythonPath, 10, SpringLayout.SOUTH, enterOutMidName);
        layout.putConstraint(SpringLayout.WEST, predict, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.SOUTH, predict, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, trainAndPredict, 5, SpringLayout.EAST, predict);
        layout.putConstraint(SpringLayout.SOUTH, trainAndPredict, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, chooseModel, 5, SpringLayout.EAST, trainAndPredict);
        layout.putConstraint(SpringLayout.SOUTH, chooseModel, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, model, 5, SpringLayout.EAST, chooseModel);
        layout.putConstraint(SpringLayout.SOUTH, model, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, getAllParamsFrom, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.SOUTH, getAllParamsFrom, -15, SpringLayout.NORTH, predict);
        layout.putConstraint(SpringLayout.WEST, paramsIndex, 5, SpringLayout.EAST, getAllParamsFrom);
        layout.putConstraint(SpringLayout.SOUTH, paramsIndex, -10, SpringLayout.NORTH, predict);
        layout.putConstraint(SpringLayout.WEST, draw, 5, SpringLayout.EAST, paramsIndex);
        layout.putConstraint(SpringLayout.SOUTH, draw, -10, SpringLayout.NORTH, predict);

        layout.putConstraint(SpringLayout.WEST, pythonPathPanel, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.EAST, pythonPathPanel, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, pythonPathPanel, 80, SpringLayout.NORTH, this);

        setLayout(layout);
        add(pythonPathPanel);
        add(pathField);
        add(loadPath);
        add(sendRandomJob);
        add(sendSteppedJob);
        add(moveLeft);
        add(moveRight);
        add(dynamicItemPanel);
        add(staticItemPanel);
        add(samplesNumber);
        add(samplesNumberField);
        add(switchToML);
        add(switchToSimulation);
        add(inputPanel);
        add(outputPanel);
        add(trainAndPredict);
        add(model);
        add(chooseModel);
        add(predict);
        add(pathMLField);
        add(loadResults);
        add(enterOutMidName);
        add(outMidName);
        add(getAllParamsFrom);
        add(paramsIndex);
        add(loadInputs);
        add(draw);
        add(showPythonPath);
        loadResults.setVisible(false);
        pathMLField.setVisible(false);
        predict.setVisible(false);
        model.setVisible(false);
        chooseModel.setVisible(false);
        trainAndPredict.setVisible(false);
        outputPanel.setVisible(false);
        inputPanel.setVisible(false);
        switchToSimulation.setVisible(false);
        enterOutMidName.setVisible(false);
        outMidName.setVisible(false);
        getAllParamsFrom.setVisible(false);
        paramsIndex.setVisible(false);
        loadInputs.setVisible(false);
        draw.setVisible(false);
        showPythonPath.setVisible(false);
        pythonPathPanel.setVisible(false);
        repaint();
    }

    public void sendRandomJob() {
        Random rand = new Random();
        for (int i = 0; i < Double.parseDouble(samplesNumberField.getText()); i++) {
            File original = new File("run.in");
            Path copied = Paths.get("run" + i + ".in");
            Path originalPath = original.toPath();
            try {
                Files.copy(originalPath, copied, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                e.printStackTrace();
            }

            ArrayList<String> lines = new ArrayList<>();
            String line;
            try {
                File f1 = new File("run" + i + ".in");
                FileReader fr = new FileReader(f1);
                BufferedReader br = new BufferedReader(fr);
                while ((line = br.readLine()) != null) {
                    for(SimulationParameter p: dynamicItemPanel.dynamicParameters){
                        if (line.contains(p.getName())) {
                            if(p.getName().contains("MAXB")) line = p.getName() + "=" + (-((p.getUpperValue()-p.getLowerValue()) * rand.nextDouble() + p.getLowerValue())*0.000588-0.0000372);
                            else line = p.getName() + "=" + ((p.getUpperValue()-p.getLowerValue()) * rand.nextDouble() + p.getLowerValue());
                        }
                    }
                    lines.add(line);
                }
                fr.close();
                br.close();

                FileWriter fw = new FileWriter(f1);
                BufferedWriter out = new BufferedWriter(fw);
                for (String s : lines) {
                    out.write(s);
                    out.write("\n");
                }
                out.flush();
                fw.close();
                out.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            sendBatch(i);
        }
        writeInputParameters();
        fileNumber = 0;
    }

    public void sendSteppedJob(int level, ArrayList<Double> values) {
        System.out.println("fileNumber: "+fileNumber);
        double currentValue = dynamicItemPanel.dynamicParameters.get(level).getLowerValue();
        while (currentValue < dynamicItemPanel.dynamicParameters.get(level).getUpperValue()) {
            if ((level + 1) < dynamicItemPanel.dynamicParameters.size()) {
                ArrayList<Double> valuesNew = new ArrayList<>(values);
                valuesNew.add(currentValue);
                sendSteppedJob(level + 1,valuesNew);
            } else {
                File original = new File("run.in");
                Path copied = Paths.get("run" + fileNumber + ".in");
                Path originalPath = original.toPath();
                try {
                    Files.copy(originalPath, copied, StandardCopyOption.REPLACE_EXISTING);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                ArrayList<String> lines = new ArrayList<>();
                String line;
                try {
                    File f1 = new File("run" + fileNumber + ".in");
                    FileReader fr = new FileReader(f1);
                    BufferedReader br = new BufferedReader(fr);
                    while ((line = br.readLine()) != null) {
                        for(int l = 0; (l+1) < dynamicItemPanel.dynamicParameters.size();l++){
                            SimulationParameter p = dynamicItemPanel.dynamicParameters.get(l);
                            if (line.contains(p.getName())) {
                                if(p.getName().contains("MAXB")) line = p.getName() + "=" + (-values.get(l)*0.000588-0.0000372);
                                else line = p.getName() + "=" + values.get(l);
                            }
                        }
                        SimulationParameter p = dynamicItemPanel.dynamicParameters.get(dynamicItemPanel.dynamicParameters.size()-1);
                        if (line.contains(p.getName())) {
                            if(p.getName().contains("MAXB")) line = p.getName() + "=" + (-currentValue*0.000588-0.0000372);
                            else line = p.getName() + "=" + currentValue;
                        }
                        lines.add(line);
                    }
                    fr.close();
                    br.close();

                    FileWriter fw = new FileWriter(f1);
                    BufferedWriter out = new BufferedWriter(fw);
                    for (String s : lines) {
                        out.write(s);
                        out.write("\n");
                    }
                    out.flush();
                    fw.close();
                    out.close();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
                sendBatch(fileNumber);
                fileNumber++;
            }
            currentValue += dynamicItemPanel.dynamicParameters.get(level).getStep();
        }
    }

    public void sendBatch(int i){
        ArrayList<String> lines = new ArrayList<>();
        String line;
        try {
            File f1 = new File("batch.sh");
            FileReader fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                if (line.contains("Astra")) {
                    line = "./Astra run" + i + ".in 2>&1 | tee run" + i + ".log";
                }
                if (line.contains("o_out")) {
                    line = "#$ -o o_out" + i + ".txt";
                }
                if (line.contains("e_out")) {
                    line = "#$ -e e_out" + i + ".txt";
                }
                lines.add(line);
            }
            fr.close();
            br.close();

            FileWriter fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (String s : lines) {
                out.write(s);
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        String s;
        Process p;
        try {
            p = Runtime.getRuntime().exec("qsub ./batch.sh");
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(p.getInputStream()));
            while ((s = br.readLine()) != null)
            p.waitFor();
            p.destroy();
        } catch (Exception ignored) {
        }
    }

    public void writeInputParameters(){
        File f1 = new File("inputParameters.txt");
        FileWriter fw;
        try {
            fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (SimulationParameter p : dynamicItemPanel.dynamicParameters) {
                out.write(p.getName());
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadList(String path){
        String line;
        try {
            File f1 = new File(path);
            FileReader fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                if (line.contains("=")) {
                    String[] parts = line.split("=");
                    if(isNumeric(parts[1])){
                        if(parts[0].contains("MAXB")) allParameters.add(new SimulationParameter(parts[0],(-Double.parseDouble(parts[1]) - 0.0000372) / 0.000588));
                        else allParameters.add(new SimulationParameter(parts[0],Double.parseDouble(parts[1])));
                    }
                }
            }
            fr.close();
            br.close();
            staticItemPanel.staticParameters = allParameters;
            DefaultTableModel dtm = new StaticTable(staticItemPanel.staticParameters);
            staticItemPanel.staticItemTable.setModel(dtm);
            dtm.addTableModelListener(staticItemPanel.staticItemTable);
            staticItemPanel.staticItemTable.setDefaultRenderer(Object.class, new ParameterTableRenderer());
            /*staticItemPanel.staticParameters = allParameters;
            ((DefaultTableModel) staticItemPanel.staticItemTable.getModel()).fireTableDataChanged();*/
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public boolean isNumeric(String strNum) {
        if (strNum == null) {
            return false;
        }
        try {
            double d = Double.parseDouble(strNum);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    public void moveSelectedLeft(){
        int[] rows = staticItemPanel.staticItemTable.getSelectedRows();
        for(int row:rows){
            staticItemPanel.staticParameters.get(row).isStatic = false;
        }
        updateStatus();
    }

    public void moveSelectedRight(){
        int[] rows = dynamicItemPanel.dynamicItemTable.getSelectedRows();
        for(int row:rows){
            dynamicItemPanel.dynamicParameters.get(row).isStatic = true;
        }
        updateStatus();
    }

    public void updateStatus(){
        dynamicItemPanel.dynamicParameters = new ArrayList<>();
        staticItemPanel.staticParameters = new ArrayList<>();
        for(SimulationParameter param: allParameters){
            if(param.isStatic) staticItemPanel.staticParameters.add(param);
            else dynamicItemPanel.dynamicParameters.add(param);
        }
        ((DynamicTable) dynamicItemPanel.dynamicItemTable.getModel()).auList = dynamicItemPanel.dynamicParameters;
        ((StaticTable) staticItemPanel.staticItemTable.getModel()).auList = staticItemPanel.staticParameters;
        ((DefaultTableModel) staticItemPanel.staticItemTable.getModel()).fireTableDataChanged();
        ((DefaultTableModel) dynamicItemPanel.dynamicItemTable.getModel()).fireTableDataChanged();
    }

    public void switchToMLAction(){
        loadResults.setVisible(true);
        pathMLField.setVisible(true);
        predict.setVisible(true);
        chooseModel.setVisible(true);
        trainAndPredict.setVisible(true);
        model.setVisible(true);
        outputPanel.setVisible(true);
        inputPanel.setVisible(true);
        switchToSimulation.setVisible(true);
        enterOutMidName.setVisible(true);
        outMidName.setVisible(true);
        getAllParamsFrom.setVisible(true);
        paramsIndex.setVisible(true);
        loadInputs.setVisible(true);
        draw.setVisible(true);
        showPythonPath.setVisible(true);
        pathField.setVisible(false);
        loadPath.setVisible(false);
        sendRandomJob.setVisible(false);
        sendSteppedJob.setVisible(false);
        moveLeft.setVisible(false);
        moveRight.setVisible(false);
        dynamicItemPanel.setVisible(false);
        staticItemPanel.setVisible(false);
        switchToML.setVisible(false);
        samplesNumber.setVisible(false);
        samplesNumberField.setVisible(false);
    }

    public void switchToSimulationAction(){
        loadResults.setVisible(false);
        pathMLField.setVisible(false);
        chooseModel.setVisible(false);
        trainAndPredict.setVisible(false);
        predict.setVisible(false);
        model.setVisible(false);
        outputPanel.setVisible(false);
        inputPanel.setVisible(false);
        switchToSimulation.setVisible(false);
        enterOutMidName.setVisible(false);
        outMidName.setVisible(false);
        getAllParamsFrom.setVisible(false);
        paramsIndex.setVisible(false);
        loadInputs.setVisible(false);
        draw.setVisible(false);
        showPythonPath.setVisible(false);
        pathField.setVisible(true);
        loadPath.setVisible(true);
        sendRandomJob.setVisible(true);
        sendSteppedJob.setVisible(true);
        moveLeft.setVisible(true);
        moveRight.setVisible(true);
        dynamicItemPanel.setVisible(true);
        staticItemPanel.setVisible(true);
        switchToML.setVisible(true);
        samplesNumber.setVisible(true);
        samplesNumberField.setVisible(true);
    }

    public void loadInputs(){
        inputPanel.parameters.clear();
        File f1 = new File("inputParameters.txt");
        FileReader fr;
        String line;
        try {
            fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                if(!line.equals("")){
                    inputPanel.parameters.add(new InputParameter(line));
                }
            }
            if(inputPanel.parameters.size()>1) {
                inputPanel.parameters.get(0).isHorizontalAxis = true;
                inputPanel.parameters.get(1).isHorizontalAxis = true;
            }
            fr.close();
            br.close();
            ((DefaultTableModel) inputPanel.table.getModel()).fireTableDataChanged();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadPath(){
        loadInputs();
        try {
            //clear?
            /*
            for(int i=1;i<130;i++) {
                Process process;
                String[] cmd = {"/afs/ifh.de/user/k/kladov/volume/pythonLocal/bin/python3.9","getInformationDESY.py", String.valueOf((10 * i)), outMidName.getText()};
                process = Runtime.getRuntime().exec(cmd);
                InputStream stdout = process.getInputStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
                String line1;
                while ((line1 = reader.readLine()) != null) {
                    System.out.println("stdout: " + line1);
                }
            }
            */
            Process process;
            String[] cmd = {pythonPathPanel.pythonPath.getText(),"getInformationDESY.py", outMidName.getText()};
            process = Runtime.getRuntime().exec(cmd);
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void correctInformation(){
        File original = new File("information.txt");
        Path copied = Paths.get("informationCorrected.txt");
        Path originalPath = original.toPath();
        try {
            Files.copy(originalPath, copied, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ArrayList<String> lines = new ArrayList<>();
        String line;
        try {
            File f1 = new File("informationCorrected.txt");
            FileReader fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                boolean isInList = true;
                for(OutputParameter p: outputPanel.parameters){
                    if(!p.estimate) {
                        if (line.contains(p.name)) {
                            isInList = false;
                            break;
                        }
                    }
                }
                if(isInList) lines.add(line);
            }
            fr.close();
            br.close();

            FileWriter fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (String s : lines) {
                out.write(s);
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void saveInputParameters(){
        File f1 = new File("inputParametersValues.txt");
        FileWriter fw;
        try {
            fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (InputParameter p: inputPanel.parameters) {
                if(p.name.contains("MAXB")) out.write(String.valueOf((-p.value*0.000588-0.0000372)));
                else out.write(String.valueOf(p.value));
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void trainAndPredict(){
        correctInformation();
        try {
            Process process;
            String[] cmd = {pythonPathPanel.pythonPath.getText(),"trainML.py", model.getSelectedItem()};
            process = Runtime.getRuntime().exec(cmd);
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        predict();
    }

    public void draw(){
        try {
            Process process;
            int xCoord = 0;
            int yCoord = 1;
            boolean wasXAssigned = false;
            for(int i = 0; i< inputPanel.parameters.size();i++){
                if(inputPanel.parameters.get(i).isHorizontalAxis) {
                    if (!wasXAssigned) {
                        xCoord = i;
                        wasXAssigned = true;
                    } else{
                        yCoord = i;
                    }
                }
            }
            int zCoord = 4;
            for(int i = 0; i< outputPanel.parameters.size();i++){
                if(outputPanel.parameters.get(i).isZAxis) {
                    zCoord = i;
                }
            }
            String[] cmd = {pythonPathPanel.pythonPath.getText(),"predict.py", model.getSelectedItem()+"_draw", String.valueOf(xCoord), String.valueOf(yCoord), String.valueOf(zCoord),paramsIndex.getText()};
            process = Runtime.getRuntime().exec(cmd);
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void predict(){
        saveInputParameters();
        try {
            Process process;
            String[] cmd = {pythonPathPanel.pythonPath.getText(),"predict.py", model.getSelectedItem()};
            process = Runtime.getRuntime().exec(cmd);
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            ArrayList<String> outputNames = new ArrayList<>();
            File f = new File("informationCorrected.txt");
            FileReader fr = new FileReader(f);
            BufferedReader br = new BufferedReader(fr);
            String line;
            boolean isOutput = false;
            int numberOfInput = 0;
            while ((line = br.readLine()) != null) {
                if(line.contains("input") && numberOfInput==1){
                    break;
                } else if(line.contains("input")){
                    numberOfInput++;
                    isOutput = false;
                } else if (line.contains("output")){
                    isOutput = true;
                } else if (isOutput){
                    String[] parts = line.split(" ");
                    outputNames.add(parts[0]);
                }
            }
            fr.close();
            br.close();

            File f1 = new File("predictOutput.txt");
            fr = new FileReader(f1);
            br = new BufferedReader(fr);
            int i = 0;
            String name;
            while ((line = br.readLine()) != null) {
                name = outputNames.get(i);
                for(OutputParameter p: outputPanel.parameters){
                    if(name.equals(p.name)){
                        p.estimatedValue = Double.parseDouble(line);
                    }
                }
                i++;
            }
            fr.close();
            br.close();
            ((OutputParameterTable) outputPanel.table.getModel()).fireTableDataChanged();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
