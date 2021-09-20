#ifndef FACEMARKLBF_H
#define FACEMARKLBF_H

#include "FacemarkLBFGlobal.h"
#include <opencv2/face.hpp>
#include "Core/C2dImageTask.h"
#include "IO/CImageIO.h"
#include "CPluginProcessInterface.hpp"

//------------------------------//
//----- CFacemarkLBFParam -----//
//------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFParam: public CWorkflowTaskParam
{
    public:

        CFacemarkLBFParam() : CWorkflowTaskParam()
        {
        }

        void setParamMap(const UMapString& paramMap) override
        {
            m_displayType = std::stoi(paramMap.at("displayType"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("displayType", std::to_string(m_displayType)));
            return map;
        }

    public:

        int m_displayType = 0;
};

//-------------------------//
//----- CFacemarkLBF -----//
//-------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBF : public C2dImageTask
{
    public:

        CFacemarkLBF();
        CFacemarkLBF(const std::string name, const std::shared_ptr<CFacemarkLBFParam>& pParam);

        size_t  getProgressSteps() override;

        void    run() override;

    private:

        // drawPolyLine draws a poly line by joining successive points between the start and end indices.
        void    drawPolyline(std::shared_ptr<CGraphicsOutput>& pOutput, std::vector<cv::Point2f> &landmarks, size_t start, size_t end, bool bIsClosed = false);
        void    drawLandmarksPoint(std::shared_ptr<CGraphicsOutput>& pOutput, std::vector<cv::Point2f> &landmarks);
        void    drawLandmarksFace(std::shared_ptr<CGraphicsOutput>& pOutput, std::vector<cv::Point2f> &landmarks);
        void    drawDelaunay(std::shared_ptr<CGraphicsOutput>& pOutput, std::vector<cv::Point2f> &landmarks);

        void    manageInputGraphics(const CMat& imgSrc);
        void    manageOutput(std::vector<std::vector<cv::Point2f>>& landmarks);

    private:

        std::vector<cv::Rect> m_faces;
        cv::Ptr<cv::face::FacemarkLBF> m_pFacemark = nullptr;
};

//--------------------------------//
//----- CFacemarkLBFFactory -----//
//--------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFFactory : public CTaskFactory
{
    public:

        CFacemarkLBFFactory()
        {
            m_info.m_name = "infer_facemark_lbf";
            m_info.m_shortDescription = QObject::tr("Facial landmark detection using Local Binary Features (LBF)").toStdString();
            m_info.m_description = QObject::tr("The locations of the fiducial facial landmark points around facial components and "
                                               "facial contour capture the rigid and non-rigid facial deformations due to head movements and facial expressions. "
                                               "They are hence important for various facial analysis tasks. ").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Face/Landmarks").toStdString();
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_keywords = "face,facial,landmark";
            m_info.m_authors = "Ren S, Cao X, Wei Y, Sun J.";
            m_info.m_article = "Face alignment at 3000 fps via regressing local binary features";
            m_info.m_journal = "CVPR";
            m_info.m_year = 2014;
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/dc/d63/classcv_1_1face_1_1FacemarkLBF.html";
            m_info.m_license = "3-clause BSD License";
            m_info.m_repo = "https://github.com/opencv/opencv";
            m_info.m_version = "1.0.0";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pFacemarkLBFParam = std::dynamic_pointer_cast<CFacemarkLBFParam>(pParam);
            if(pFacemarkLBFParam != nullptr)
                return std::make_shared<CFacemarkLBF>(m_info.m_name, pFacemarkLBFParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pFacemarkLBFParam = std::make_shared<CFacemarkLBFParam>();
            assert(pFacemarkLBFParam != nullptr);
            return std::make_shared<CFacemarkLBF>(m_info.m_name, pFacemarkLBFParam);
        }
};

//-------------------------------//
//----- CFacemarkLBFWidget -----//
//-------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFWidget: public CWorkflowTaskWidget
{
    public:

        CFacemarkLBFWidget(QWidget *parent = Q_NULLPTR): CWorkflowTaskWidget(parent)
        {
            init();
        }
        CFacemarkLBFWidget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<CFacemarkLBFParam>(pParam);
            init();
        }

    private:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CFacemarkLBFParam>();

            m_pCombo = addCombo(0, tr("Display type"));
            m_pCombo->addItem("Points", 0);
            m_pCombo->addItem("Face", 1);
            m_pCombo->addItem("Delaunay", 2);
            m_pCombo->setCurrentIndex(m_pCombo->findData(m_pParam->m_displayType));
        }

        void onApply() override
        {
            m_pParam->m_displayType = m_pCombo->currentData().toInt();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<CFacemarkLBFParam>  m_pParam = nullptr;
        QComboBox*  m_pCombo;
};

//--------------------------------------//
//----- CFacemarkLBFWidgetFactory -----//
//--------------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFWidgetFactory : public CWidgetFactory
{
    public:

        CFacemarkLBFWidgetFactory()
        {
            m_name = "infer_facemark_lbf";
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CFacemarkLBFWidget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CFacemarkLBFFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CFacemarkLBFWidgetFactory>();
        }
};

#endif // FACEMARKLBF_H
